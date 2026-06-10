__author__ = "Paras Sharma"

import numpy as np
from scipy.interpolate import interp1d
from slsim.Microlensing.source_morphology.source_morphology import SourceMorphology
import slsim.ImageSimulation.image_quality_lenstronomy as iql


class SupernovaeSourceMorphology(SourceMorphology):
    """Class for Supernovae source morphology."""

    def __init__(
        self,
        observing_wavelength_band,
        source_redshift,
        cosmo,
        sn_model_instance=None,  # RandomizedSupernova/Supernova instance — IS a sncosmo.Model
        sn_modeldir=None,
        sn_model_name="hsiao",
        ellipticity=1.0,
        grid_pixels=300,
        v_base_km_s=10000.0,
        v_uv_km_s=18000.0,
        u_limb_uv=0.8,
        u_limb_ir=0.2,
        anchor_spacing_days=5.0,
        user_snapshots=None,
        *args,
        **kwargs,
    ):
        """
        :param observing_wavelength_band: e.g., 'g', 'r', 'i', or a sncosmo bandpass name.
        :param source_redshift: Redshift of the SN.
        :param cosmo: Astropy cosmology instance.
        :param sn_modeldir: Directory containing sncosmo model files.   
        :param sn_model_name: sncosmo template name (default 'hsiao').
        :param ellipticity: Asymmetry of the explosion (default 1.0).
        :param grid_pixels: Resolution of the kernel map.
        :param v_base_km_s: Velocity of the optical/IR photosphere in km/s.
        :param v_uv_km_s: Velocity of the UV line-blanketed shell in km/s.
        :param u_limb_uv: Limb darkening coefficient at 3000 Å.
        :param u_limb_ir: Limb darkening coefficient at 10000 Å.
        :param anchor_spacing_days: Source-frame days between analytical
            evaluations. Controls the speed/accuracy tradeoff. Default is 5.0.
        :param user_snapshots: Optional dict containing pre-computed snapshots for time-varying sources.
            If provided, sncosmo analytical calculations are completely bypassed.
            Must contain:
            - 'times': 1D array of source-frame times (in days).
            - 'kernels': List or 3D array of 2D kernel maps normalized to 1.
            - 'pixel_scales_m': 1D array of pixel scales in meters corresponding to each kernel.
        """
        if observing_wavelength_band in iql.get_all_supported_bands():
            self.band = iql.get_sncosmo_filtername(observing_wavelength_band)
        else:
            self.band = observing_wavelength_band

        self.source_redshift = source_redshift
        self.cosmo = cosmo
        self.sn_model_instance = sn_model_instance
        self.sn_model_name = sn_model_name
        self.sn_modeldir = sn_modeldir
        self.ellipticity = ellipticity

        self.v_base_km_s = v_base_km_s
        self.v_uv_km_s = v_uv_km_s
        self.u_limb_uv = u_limb_uv
        self.u_limb_ir = u_limb_ir

        self._num_pix_x = grid_pixels
        self._num_pix_y = grid_pixels

        # Only compute snapshots if the user didn't provide custom snapshots. This allows for maximum flexibility and reuse of expensive computations.
        if user_snapshots is None:
            user_snapshots = self._build_analytical_snapshots(anchor_spacing_days)

        # Hand off the snapshots to the Base Class for vectorized interpolation
        super().__init__(
            is_time_varying=True, user_snapshots=user_snapshots, *args, **kwargs
        )

    def _build_analytical_snapshots(self, anchor_spacing_days):
        """Internal helper to load sncosmo and generate the analytical anchors.

        Bypassed if user provides their own grids.

        :param anchor_spacing_days: Source-frame days between analytical
            evaluations. Controls the speed/accuracy tradeoff.
        :return: Dictionary containing 'times', 'kernels', and
            'pixel_scales_m' for the analytical snapshots.
        """
        import sncosmo
        if self.sn_model_instance is not None:
            # RandomizedSupernova IS a sncosmo.Model — use directly.
            # x1 and c are already set by RandomizedSupernova.__init__,
            # so spectral shape matches the lightcurve exactly.
            self._sn_model = self.sn_model_instance

        elif self.sn_modeldir is not None:
            source = sncosmo.SALT3Source(modeldir=self.sn_modeldir)
            self._sn_model = sncosmo.Model(source=source)
            self._sn_model.set(x1=0.0, c=0.0)  # average SN Ia; shape is all that matters

        else:
            self._sn_model = sncosmo.Model(source=self.sn_model_name)

        try:
            self._bandpass = sncosmo.get_bandpass(self.band)
        except Exception:
            raise ValueError(f"Band {self.band} not recognized by sncosmo.")

        # Generate the analytical anchors automatically
        min_t = self._sn_model.mintime()
        max_t = self._sn_model.maxtime()
        anchor_times = np.arange(
            min_t, max_t + anchor_spacing_days, anchor_spacing_days
        )

        kernels, pixel_scales_m = self._generate_analytical_anchors(anchor_times)

        return {
            "times": anchor_times,
            "kernels": kernels,
            "pixel_scales_m": pixel_scales_m,
        }

    def _continuous_monochromatic_morphology(
        self, wavelength_angstroms, time_seconds, R_eff
    ):
        """Generates the continuous wavelength-dependent spatial profile.
        Evaluated on a 1D radial grid for O(N) optimization.

        :param wavelength_angstroms: Wavelength in Angstroms.
        :param time_seconds: Time since explosion in seconds.
        :param R_eff: 1D array of effective radii in meters at which to
            evaluate the profile.
        :return: 1D array of intensity values corresponding to R_eff.
        """
        v_base_m_s = self.v_base_km_s * 1000.0
        v_uv_m_s = self.v_uv_km_s * 1000.0

        opacity_proxy = np.exp(-(wavelength_angstroms - 3000) / 2000)
        v_phot_m_s = v_base_m_s + (v_uv_m_s - v_base_m_s) * opacity_proxy
        r_phot = max(v_phot_m_s * time_seconds, 1e8)

        u_limb_diff = self.u_limb_uv - self.u_limb_ir
        u_limb = self.u_limb_uv - u_limb_diff * ((wavelength_angstroms - 3000) / 7000)
        u_min, u_max = min(self.u_limb_uv, self.u_limb_ir), max(
            self.u_limb_uv, self.u_limb_ir
        )
        u_limb = np.clip(u_limb, u_min, u_max)

        intensity = np.zeros_like(R_eff)
        mask = R_eff <= r_phot

        if not np.any(mask):
            return intensity

        R_eff_masked = R_eff[mask]
        mu = np.sqrt(1.0 - (R_eff_masked / r_phot) ** 2)

        major_lines = [(3934, 100), (6355, 150), (8542, 150)]
        ring_fraction = 0.0
        for line_wave, line_width in major_lines:
            dist = abs(wavelength_angstroms - line_wave)
            if dist < line_width:
                ring_fraction = max(
                    ring_fraction, np.exp(-((dist / (line_width / 2)) ** 2))
                )

        continuum_intensity = 1.0 - u_limb * (1.0 - mu)
        ring_intensity = (R_eff_masked / r_phot) ** 2 * np.exp(
            -((R_eff_masked / (r_phot * 0.8)) ** 4)
        )

        intensity[mask] = (
            1.0 - ring_fraction
        ) * continuum_intensity + ring_fraction * ring_intensity
        return intensity

    def get_kernel_map(self, time_days):
        """Builds the SED-weighted kernel map using O(N) 1D radial integration
        followed by an exact 2D projection."""
        # Offset time so t=0 is the explosion, not the peak
        time_since_explosion_days = time_days - self._sn_model.mintime()

        # Prevent negative times if the array happens to probe before the explosion
        time_since_explosion_days = max(time_since_explosion_days, 0.0)
        time_seconds = time_since_explosion_days * 24 * 3600
        max_v_m_s = max(self.v_base_km_s, self.v_uv_km_s) * 1000.0
        r_max_meters = max(max_v_m_s * time_seconds, 1e8)
        max_scale_m = r_max_meters * 2.5

        wavelengths = self._bandpass.wave
        transmissions = self._bandpass.trans
        sed_flux = self._sn_model.flux(time_days, wavelengths)

        valid_mask = (transmissions >= 0.01) & (sed_flux > 0)
        valid_waves = wavelengths[valid_mask]
        valid_weights = (sed_flux * transmissions)[valid_mask]

        # OPTIMIZATION: 1D Radial Integration
        # We integrate out to sqrt(2) * max_scale_m to cover the corners of the 2D box
        r_1d = np.linspace(0, max_scale_m * np.sqrt(2), 1000)
        intensity_1d = np.zeros_like(r_1d)

        for wave, weight in zip(valid_waves, valid_weights):
            spatial_profile = self._continuous_monochromatic_morphology(
                wave, time_seconds, r_1d
            )
            intensity_1d += spatial_profile * weight

        # Create the fast 1D to 2D interpolator
        interpolator = interp1d(
            r_1d, intensity_1d, kind="linear", bounds_error=False, fill_value=0.0
        )

        # Broadcast to 2D
        x = np.linspace(-max_scale_m, max_scale_m, self._num_pix_x)
        y = np.linspace(-max_scale_m, max_scale_m, self._num_pix_y)
        X, Y = np.meshgrid(x, y)
        R_eff_2D = np.sqrt(X**2 + (Y / self.ellipticity) ** 2)

        kernel = interpolator(R_eff_2D)

        if np.nansum(kernel) > 0:
            kernel /= np.nansum(kernel)

        current_pixel_scale_m = (2.0 * max_scale_m) / self._num_pix_x
        return kernel, current_pixel_scale_m

    def _generate_analytical_anchors(self, time_anchors_days):
        """Internal helper to generate the sparse anchors during
        initialization.

        :param time_anchors_days: 1D array of source-frame times (in
            days) at which to generate the analytical kernels.
        :return: Tuple of (kernels, pixel_scales_m) where kernels is a
            list of 2D kernel maps and pixel_scales_m is a list of
            corresponding pixel scales in
        """
        kernels = []
        pixel_scales_m = []
        for t in time_anchors_days:
            kernel, p_scale = self.get_kernel_map(time_days=t)
            kernels.append(kernel)
            pixel_scales_m.append(p_scale)
        return kernels, pixel_scales_m
