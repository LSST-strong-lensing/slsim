__author__ = "Paras Sharma"

import numpy as np
import sncosmo
from slsim.Microlensing.source_morphology.source_morphology import SourceMorphology
import slsim.ImageSimulation.image_quality_lenstronomy as iql


class SupernovaeSourceMorphology(SourceMorphology):
    """Class for Supernovae source morphology."""

    def __init__(
        self,
        observing_wavelength_band,
        source_redshift,
        cosmo,
        sn_model_name="hsiao",
        ellipticity=1.0,
        grid_pixels=300,
        v_base_km_s=10000.0,
        v_uv_km_s=18000.0,
        u_limb_uv=0.8,
        u_limb_ir=0.2,
        *args, **kwargs
    ):
        """
        :param observing_wavelength_band: e.g., 'g', 'r', 'i', or a sncosmo bandpass name.
        :param source_redshift: Redshift of the SN.
        :param cosmo: Astropy cosmology instance.
        :param sn_model_name: sncosmo template name (default 'hsiao').
        :param ellipticity: Asymmetry of the explosion (default 1.0).
        :param grid_pixels: Resolution of the kernel map.
        :param v_base_km_s: Velocity of the optical/IR photosphere in km/s (default 10000.0).
        :param v_uv_km_s: Velocity of the UV line-blanketed shell in km/s (default 18000.0).
        :param u_limb_uv: Limb darkening coefficient at 3000 Å (default 0.8).
        :param u_limb_ir: Limb darkening coefficient at 10000 Å (default 0.2).
        """
        super().__init__(is_time_varying=True, *args, **kwargs)

        # band can be slsim convention or sncosmo bandpass name
        if observing_wavelength_band in iql.get_all_supported_bands():
            self.band = iql.get_sncosmo_filtername(observing_wavelength_band)
        else:
            self.band = observing_wavelength_band

        self.source_redshift = source_redshift
        self.cosmo = cosmo
        self.sn_model_name = sn_model_name
        self.ellipticity = ellipticity
        
        self.v_base_km_s = v_base_km_s
        self.v_uv_km_s = v_uv_km_s
        self.u_limb_uv = u_limb_uv
        self.u_limb_ir = u_limb_ir

        self._num_pix_x = grid_pixels
        self._num_pix_y = grid_pixels
        self._sn_model = sncosmo.Model(source=self.sn_model_name)

    def _continuous_monochromatic_morphology(self, wavelength_angstroms, time_seconds, max_scale_m):
        """Generates the continuous wavelength-dependent spatial profile."""
        x = np.linspace(-max_scale_m, max_scale_m, self._num_pix_x)
        y = np.linspace(-max_scale_m, max_scale_m, self._num_pix_y)
        X, Y = np.meshgrid(x, y)

        R_eff = np.sqrt(X**2 + (Y / self.ellipticity) ** 2)

        # Chromatic Velocity Interpolation (m/s)
        v_base_m_s = self.v_base_km_s * 1000.0
        v_uv_m_s = self.v_uv_km_s * 1000.0
        
        # Exponential proxy for the opacity wall transition
        opacity_proxy = np.exp(-(wavelength_angstroms - 3000) / 2000)
        v_phot_m_s = v_base_m_s + (v_uv_m_s - v_base_m_s) * opacity_proxy

        # Calculate actual physical radius (R = v * t)
        r_phot = max(v_phot_m_s * time_seconds, 1e8)

        # Wavelength-Dependent Limb Darkening
        u_limb_diff = self.u_limb_uv - self.u_limb_ir
        u_limb = self.u_limb_uv - u_limb_diff * ((wavelength_angstroms - 3000) / 7000)
        
        # Safely clip to user bounds regardless of order
        u_min, u_max = min(self.u_limb_uv, self.u_limb_ir), max(self.u_limb_uv, self.u_limb_ir)
        u_limb = np.clip(u_limb, u_min, u_max)
        
        mu = np.sqrt(1.0 - np.clip((R_eff / r_phot) ** 2, 0, 1))

        # Ring Detection (Absorption lines)
        major_lines = [(3934, 100), (6355, 150), (8542, 150)]
        ring_fraction = 0.0
        for line_wave, line_width in major_lines:
            dist = abs(wavelength_angstroms - line_wave)
            if dist < line_width:
                ring_fraction = max(
                    ring_fraction, np.exp(-((dist / (line_width / 2)) ** 2))
                )

        continuum_intensity = 1.0 - u_limb * (1.0 - mu)
        ring_intensity = (R_eff / r_phot) ** 2 * np.exp(
            -((R_eff / (r_phot * 0.8)) ** 4)
        )

        intensity = (
            1.0 - ring_fraction
        ) * continuum_intensity + ring_fraction * ring_intensity
        intensity[R_eff > r_phot] = 0.0

        return intensity

    def get_kernel_map(self, time_days):
        """Builds the SED-weighted kernel map for a specific epoch.
        Returns the kernel and its physical pixel scale in meters.
        """
        time_seconds = time_days * 24 * 3600

        # Define grid size based on the maximum possible expansion velocity
        max_v_m_s = max(self.v_base_km_s, self.v_uv_km_s) * 1000.0
        r_max_meters = max(max_v_m_s * time_seconds, 1e8)
        
        # Grid width is 2.5x the maximum radius on each side
        max_scale_m = r_max_meters * 2.5 

        try:
            bandpass = sncosmo.get_bandpass(self.band)
        except Exception:
            raise ValueError(f"Band {self.band} not recognized by sncosmo.")

        wavelengths = bandpass.wave
        transmissions = bandpass.trans
        sed_flux = self._sn_model.flux(time_days, wavelengths)

        kernel = np.zeros((self._num_pix_x, self._num_pix_y))

        # SED Integration
        for wave, flux, trans in zip(wavelengths, sed_flux, transmissions):
            if trans < 0.01 or flux <= 0:
                continue
            spatial_profile = self._continuous_monochromatic_morphology(
                wave, time_seconds, max_scale_m
            )
            kernel += spatial_profile * flux * trans

        if np.nansum(kernel) > 0:
            kernel /= np.nansum(kernel)

        # Total width is 2 * max_scale_m. Divide by num pixels to get meters/pixel.
        current_pixel_scale_m = (2.0 * max_scale_m) / self._num_pix_x

        return kernel, current_pixel_scale_m

    def get_time_dependent_kernel_maps(self, time_anchors_days):
        """Generates a sequence of evolving kernels for the supernova.

        Returns the kernels and their corresponding physical pixel scale
        in meters.
        """
        kernels = []
        pixel_scales_m = []

        for t in time_anchors_days:
            kernel, p_scale = self.get_kernel_map(time_days=t)
            kernels.append(kernel)
            pixel_scales_m.append(p_scale)

        return kernels, pixel_scales_m