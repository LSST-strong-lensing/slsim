__author__ = "Paras Sharma"

import numpy as np

from slsim.Util.astro_util import (
    calculate_accretion_disk_emission,
    calculate_gravitational_radius,
)

from astropy import units as u

import speclite.filters

from slsim.Microlensing.source_morphology.source_morphology import (
    SourceMorphology,
)

from slsim.ImageSimulation.image_quality_lenstronomy import get_speclite_filtername


class AGNSourceMorphology(SourceMorphology):
    """Class for AGN (accretion disk) source morphology.

    Supports two modes:

    **Static mode** (default, ``is_time_varying=False``, no ``user_snapshots``):
        A single emission map is computed once at the observer-frame wavelength
        and cached as ``kernel_map``.  Pixel scales are derived analytically
        from the accretion-disk size parameters, so no 2-D array needs to exist
        before the scales are known.

    **Time-varying mode via external snapshots** (``user_snapshots`` provided):
        The caller supplies a pre-computed sequence of kernel maps together with
        their physical pixel scales.  The base-class vectorised interpolator
        handles temporal queries through ``get_time_dependent_kernel_maps()``.
        The static ``kernel_map`` property raises ``AttributeError`` in this mode
        (use ``get_time_dependent_kernel_maps()`` instead).
        Pixel scales are taken directly from the ``user_snapshots`` dictionary
        rather than being derived analytically, because the physical size of the
        source may evolve over time.

    Note
    ----
    Analytical time-varying AGN generation (without external grids) is not yet
    implemented and will raise ``NotImplementedError``.
    """

    def __init__(
        self,
        source_redshift,
        cosmo,
        r_out=1000,
        r_resolution=1000,
        black_hole_mass_exponent=8.0,
        inclination_angle=0,
        black_hole_spin=0,  # Spin of the black hole
        observer_frame_wavelength_in_nm=600,  # Wavelength in nanometers used to determine black body flux.
        eddington_ratio=0.15,  # Eddington ratio of the accretion disk
        observing_wavelength_band: str = None,
        is_time_varying=False,
        user_snapshots=None,
        *args,
        **kwargs,
    ):
        """Initializes the AGN source morphology.

        :param source_redshift: Redshift of the source
        :param cosmo: Astropy cosmology object for angle calculations
        :param r_out: Outer radius of the accretion disk in
            gravitational radii. This typically can be chosen as 10^3 to
            10^5 [R_g] Default is 1000.
        :param r_resolution: Resolution of the accretion disk in
            gravitational radii. Default is 1000.
        :param black_hole_mass_exponent: Exponent of the mass of the supermassive
            black hole in kg. Default is 8.0.
        :param inclination_angle: Inclination angle of the disk in
            degrees. Default is 0.
        :param black_hole_spin: The dimensionless spin parameter of the
            black hole, where the spinless case (spin = 0) corresponds
            to a Schwarzschild black hole. Positive spin represents the
            accretion disk's angular momentum is aligned with the black
            hole's spin, and negative spin represents retrograde
            accretion flow.
        :param observer_frame_wavelength_in_nm: Wavelength in nanometers
            used to determine black body flux. For the surface flux
            density of the AccretionDisk at desired wavelength. Default
            is 600 nm. This can be set to None if the
            observing_wavelength_band is provided.
        :param observing_wavelength_band: Wavelength band for the source morphology.
            Default is None. Options are:

            - LSST: "u", "g", "r", "i", "z", "y"

            If None, the observer_frame_wavelength_in_nm is used and the kernel map is generated for that wavelength.
            If a wavelength band is provided, the kernel map is generated at the mean wavelength of the band.
        :param eddington_ratio: Eddington ratio of the accretion disk.
            Default is 0.15.
        :param is_time_varying: Boolean flag indicating if the AGN source varies
            temporally. Default is False.
        :param user_snapshots: Optional dictionary containing pre-computed
            snapshots for time-varying AGN morphologies. If provided, the Base
            Class automatically handles vectorized temporal interpolation.
            Must contain:
            - 'times': 1D array of source-frame times (in days).
            - 'kernels': List or 3D array of 2D kernel maps normalized to 1.
            - 'pixel_scales_m': 1D array of pixel scales in meters corresponding to each kernel.
        """
        self.source_redshift = source_redshift
        self.cosmo = cosmo
        self.r_out = r_out
        self.r_resolution = r_resolution
        self.black_hole_mass_exponent = black_hole_mass_exponent
        self.inclination_angle = inclination_angle
        self.black_hole_spin = black_hole_spin
        self.observer_frame_wavelength_in_nm = observer_frame_wavelength_in_nm
        self.observing_wavelength_band = observing_wavelength_band
        self.eddington_ratio = eddington_ratio
        self.black_hole_mass = 10**black_hole_mass_exponent

        # Assign the observer frame wavelength in nm if the band is provided
        if self.observing_wavelength_band is not None:
            filter_obj = speclite.filters.load_filter(
                get_speclite_filtername(self.observing_wavelength_band)
            )
            self.observer_frame_wavelength_in_nm = filter_obj.effective_wavelength.to(
                u.nm
            ).value
            # TODO: In future handle this by integrating the flux map over the band

        # Generate analytical snapshots if requested but not provided
        if is_time_varying and user_snapshots is None:
            user_snapshots = self._build_analytical_snapshots()

        # Hand off to the Base Class
        super().__init__(
            is_time_varying=is_time_varying,
            user_snapshots=user_snapshots,
            *args,
            **kwargs,
        )

        self._setup_metadata()

    def _build_analytical_snapshots(self):
        """Internal helper to generate analytical time-varying anchors.

        Currently not implemented for AGN. Bypassed if user provides
        their own grids.
        """
        raise NotImplementedError(
            "Time-varying AGN source morphology is currently only supported "
            "via the injection of external grids into 'user_snapshots'. "
            "Analytical time-varying generation is not yet implemented."
        )

    def _setup_metadata(self):
        """Sets up the required pixel scale and resolution attributes based on
        the run mode."""
        if self.is_time_varying:
            # Pixel scales come from the snapshots. Use the first entry as a
            # representative value for metadata; actual per-step scales are
            # supplied by the base-class interpolator.
            rep_pixel_scale_m = float(self.user_snapshots["pixel_scales_m"][0])
            self._pixel_scale_x_m = rep_pixel_scale_m
            self._pixel_scale_y_m = rep_pixel_scale_m
            self._pixel_scale_m = rep_pixel_scale_m

            # Representative num_pix from the first kernel shape.
            first_kernel = np.asarray(self.user_snapshots["kernels"][0])
            self._num_pix_y = first_kernel.shape[0]
            self._num_pix_x = first_kernel.shape[1]

        else:
            # Static AGN: the emission map shape is (2*r_resolution, 2*r_resolution).
            # This is determined analytically — no need to call kernel_map here.
            gravitational_radius_of_smbh = calculate_gravitational_radius(
                self.black_hole_mass_exponent
            )
            num_pix = 2 * self.r_resolution
            kernel_pixel_size_m = (
                (2 * self.r_out * gravitational_radius_of_smbh / num_pix).to(u.m).value
            )

            self._pixel_scale_x_m = kernel_pixel_size_m
            self._pixel_scale_y_m = kernel_pixel_size_m
            self._pixel_scale_m = kernel_pixel_size_m

            self._num_pix_x = num_pix
            self._num_pix_y = num_pix

        # Convert pixel scales to arcseconds
        self._pixel_scale_x = self.metres_to_arcsecs(
            self._pixel_scale_x_m, self.cosmo, self.source_redshift
        )
        self._pixel_scale_y = self.metres_to_arcsecs(
            self._pixel_scale_y_m, self.cosmo, self.source_redshift
        )
        self._pixel_scale = np.sqrt(self._pixel_scale_x * self._pixel_scale_y)

        self._length_x = self._num_pix_x * self._pixel_scale_x
        self._length_y = self._num_pix_y * self._pixel_scale_y

    def get_kernel_map(self):
        """Compute and return the normalised 2-D AGN emission-map kernel.

        The map is calculated at the (rest-frame) wavelength corresponding to
        ``observer_frame_wavelength_in_nm`` corrected for redshift, using the
        thin-disk SED model in ``calculate_accretion_disk_emission``.

        Returns
        -------
        numpy.ndarray
            2-D array of shape ``(2*r_resolution, 2*r_resolution)`` normalised
            so that its sum equals 1.
        """
        # Convert observer-frame wavelength to rest-frame.
        rest_frame_wavelength_nm = self.observer_frame_wavelength_in_nm / (
            1 + self.source_redshift
        )

        emission_map = calculate_accretion_disk_emission(
            r_out=self.r_out,
            r_resolution=self.r_resolution,
            inclination_angle=self.inclination_angle,
            rest_frame_wavelength_in_nanometers=rest_frame_wavelength_nm,
            black_hole_mass_exponent=self.black_hole_mass_exponent,
            black_hole_spin=self.black_hole_spin,
            eddington_ratio=self.eddington_ratio,
            return_spectral_radiance_distribution=True,
        )
        emission_map = np.asarray(emission_map, dtype=float)

        total = np.nansum(emission_map)
        if total > 0:
            emission_map /= total

        return emission_map
