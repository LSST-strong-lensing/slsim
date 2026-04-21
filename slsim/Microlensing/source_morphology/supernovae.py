__author__ = "Paras Sharma"

import numpy as np
import sncosmo
from slsim.Microlensing.source_morphology.source_morphology import SourceMorphology
import slsim.ImageSimulation.image_quality_lenstronomy as iql


class SupernovaeSourceMorphology(SourceMorphology):
    """Class for Supernovae source morphology."""

    def __init__(
        self,
        time_days,
        observing_wavelength_band,
        source_redshift,
        cosmo,
        sn_model_name="hsiao",
        ellipticity=1.0,
        grid_pixels=300,
    ):
        """
        :param time_days: Time since explosion in days (source frame).
        :param observing_wavelength_band: e.g., 'g', 'r', 'i', or a sncosmo bandpass name.
        :param source_redshift: Redshift of the SN.
        :param cosmo: Astropy cosmology instance.
        :param sn_model_name: sncosmo template name (default 'hsiao').
        :param ellipticity: Asymmetry of the explosion (default 1.0).
        :param grid_pixels: Resolution of the kernel map.
        """
        super().__init__()
        self.time_days = time_days

        # band can be slsim convention or sncosmo bandpass name
        if observing_wavelength_band in iql.get_all_supported_bands():
            self.band = iql.get_sncosmo_filtername(observing_wavelength_band)
        else:
            self.band = observing_wavelength_band

        self.source_redshift = source_redshift
        self.cosmo = cosmo
        self.sn_model_name = sn_model_name
        self.ellipticity = ellipticity

        self._num_pix_x = grid_pixels
        self._num_pix_y = grid_pixels
        self._sn_model = sncosmo.Model(source=self.sn_model_name)

    def _continuous_monochromatic_morphology(self, wavelength_angstroms, r_base_meters):
        """Generates the continuous wavelength-dependent spatial profile."""
        # Grid width is 2.5x the base radius on each side (total width = 5x r_base)
        max_scale_m = r_base_meters * 2.5

        x = np.linspace(-max_scale_m, max_scale_m, self._num_pix_x)
        y = np.linspace(-max_scale_m, max_scale_m, self._num_pix_y)
        X, Y = np.meshgrid(x, y)

        R_eff = np.sqrt(X**2 + (Y / self.ellipticity) ** 2)

        # Chromatic Radius
        opacity_proxy = np.exp(-(wavelength_angstroms - 3000) / 2000)
        r_phot = r_base_meters * (0.8 + 0.5 * max(0, opacity_proxy))

        # Limb Darkening
        u_limb = 0.8 - 0.6 * ((wavelength_angstroms - 3000) / 7000)
        u_limb = np.clip(u_limb, 0.2, 0.8)
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

        return intensity, max_scale_m

    def get_kernel_map(self):
        """Builds the SED-weighted kernel map and sets the current physical
        pixel scale."""
        v_base_m_s = 1.0e7  # 10,000 km/s in m/s from Dobler & Keeton (2006)

        # Ensure radius is non-zero at very early times
        r_base_meters = max(v_base_m_s * (self.time_days * 24 * 3600), 1e8)

        try:
            bandpass = sncosmo.get_bandpass(self.band)
        except Exception:
            raise ValueError(f"Band {self.band} not recognized by sncosmo.")

        wavelengths = bandpass.wave
        transmissions = bandpass.trans
        sed_flux = self._sn_model.flux(self.time_days, wavelengths)

        kernel = np.zeros((self._num_pix_x, self._num_pix_y))
        current_max_scale_m = 0

        # SED Integration
        for wave, flux, trans in zip(wavelengths, sed_flux, transmissions):
            if trans < 0.01 or flux <= 0:
                continue
            spatial_profile, max_scale_m = self._continuous_monochromatic_morphology(
                wave, r_base_meters
            )
            kernel += spatial_profile * flux * trans
            current_max_scale_m = max_scale_m  # They all use the same grid boundary

        if np.nansum(kernel) > 0:
            kernel /= np.nansum(kernel)

        # Total width is 2 * max_scale_m. Divide by num pixels to get meters/pixel.
        self._current_pixel_scale_m = (2.0 * current_max_scale_m) / self._num_pix_x

        return kernel

    def get_time_dependent_kernel_maps(self, time_anchors_days):
        """Generates a sequence of evolving kernels for the supernova.

        Returns the kernels and their corresponding physical pixel scale
        in meters.
        """
        kernels = []
        pixel_scales_m = []

        for t in time_anchors_days:
            self.time_days = t
            kernel = self.get_kernel_map()
            kernels.append(kernel)
            pixel_scales_m.append(self._current_pixel_scale_m)

        return kernels, pixel_scales_m
