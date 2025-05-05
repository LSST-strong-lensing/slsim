__author__ = "Paras Sharma"

# This file contains morphologies for various types of sources.
# Currently, it contains the following morphologies:
# 1. Point source as a Gaussian
# 2. AGN Disk simulated at different wavelengths and inclination angles
# 3. Supernovae (#TODO: Needs to be implemented in future!)

import numpy as np

from slsim.Util.astro_util import (
    calculate_accretion_disk_emission,
    calculate_gravitational_radius,
)

from astropy import units as u

import speclite.filters


class SourceMorphology:
    """Base class for source morphologies."""

    def __init__(self, *args, **kwargs):
        pass

    def get_kernel_map(self, *args, **kwargs):
        """Returns the 2D array of the kernel map.

        The kernel map is a 2D array that represents the morphology of
        the source. The kernel map is used to convolve with the
        microlensing magnification map.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def kernel_map(self):
        """Returns the 2D array of the kernel map.

        The kernel map is a 2D array that represents the morphology of
        the source. The kernel map is used to convolve with the
        microlensing magnification map.
        """
        if not hasattr(self, "_kernel_map"):
            self._kernel_map = self.get_kernel_map()
        return self._kernel_map

    @property
    def length_x(self):
        """Returns the length of the 2D kernel map in x direction in
        arcseconds."""
        return self._length_x

    @property
    def length_y(self):
        """Returns the length of the 2D kernel map in y direction in
        arcseconds."""
        return self._length_y

    @property
    def num_pix_x(self):
        """Returns the number of pixels in x direction."""
        if not hasattr(self, "_num_pix_x"):
            self._num_pix_x = np.size(self.kernel_map, 0)
        return self._num_pix_x

    @property
    def num_pix_y(self):
        """Returns the number of pixels in y direction."""
        if not hasattr(self, "_num_pix_y"):
            self._num_pix_y = np.size(self.kernel_map, 1)
        return self._num_pix_y

    @property
    def pixel_scale_x(self):
        """Returns the pixel scale in x direction in arcseconds."""
        return self._pixel_scale_x

    @property
    def pixel_scale_y(self):
        """Returns the pixel scale in y direction in arcseconds."""
        return self._pixel_scale_y

    @property
    def pixel_scale(self):
        """Returns the geometric mean pixel scale in arcseconds."""
        if not hasattr(self, "_pixel_scale"):
            if hasattr(self, "_pixel_scale_x") and hasattr(self, "_pixel_scale_y"):
                self._pixel_scale = np.sqrt(self._pixel_scale_x * self._pixel_scale_y)
            else:
                raise AttributeError("Pixel scale not defined.")
        return self._pixel_scale

    @property
    def pixel_scale_x_m(self):
        """Returns the pixel scale in x direction in meters."""
        if not hasattr(self, "_pixel_scale_x_m"):
            self._pixel_scale_x_m = self.arcsecs_to_metres(
                self.pixel_scale_x, self.cosmo, self.source_redshift
            )
        return self._pixel_scale_x_m

    @property
    def pixel_scale_y_m(self):
        """Returns the pixel scale in y direction in meters."""
        if not hasattr(self, "_pixel_scale_y_m"):
            self._pixel_scale_y_m = self.arcsecs_to_metres(
                self.pixel_scale_y, self.cosmo, self.source_redshift
            )
        return self._pixel_scale_y_m

    @property
    def pixel_scale_m(self):
        """Returns the geometric mean pixel scale in meters."""
        if not hasattr(self, "_pixel_scale_m"):
            self._pixel_scale_m = self.arcsecs_to_metres(
                self.pixel_scale, self.cosmo, self.source_redshift
            )
        return self._pixel_scale_m

    def arcsecs_to_metres(self, arcsecs, cosmo, redshift):
        """Converts arcseconds to meters in the source plane, given the
        cosmology and redshift.

        :param arcsecs: Arcseconds to be converted.
        :param cosmo: Astropy cosmology object for angle calculations.
        :param redshift: Redshift of the source.
        :return: Transverse distance in meters in the source plane at
            the given redshift.
        """
        # Convert arcseconds to radians
        radians = arcsecs * u.arcsec.to(u.rad)
        # Calculate the angular diameter distance in meters
        angular_diameter_distance = (
            cosmo.angular_diameter_distance(redshift).to(u.m)
        ).value
        # Calculate the transverse distance in meters
        transverse_distance = angular_diameter_distance * radians
        return transverse_distance

    def metres_to_arcsecs(self, metres, cosmo, redshift):
        """Converts meters to arcseconds in the source plane, given the
        cosmology and redshift.

        :param metres: Meters to be converted.
        :param cosmo: Astropy cosmology object for angle calculations.
        :param redshift: Redshift of the source.
        :return: Arcseconds in the source plane at the given redshift.
        """
        # Calculate the angular diameter distance in meters
        angular_diameter_distance = (
            cosmo.angular_diameter_distance(redshift).to(u.m).value
        )
        # Calculate the arcseconds in the source plane
        arcsecs = (metres / angular_diameter_distance) * u.rad.to(u.arcsec)  # .value
        return arcsecs


class GaussianSourceMorphology(SourceMorphology):
    """Class for Gaussian source morphology."""

    def __init__(
        self,
        source_redshift,
        cosmo,
        source_size,
        length_x,
        length_y,
        num_pix_x,
        num_pix_y,
        center_x=0,
        center_y=0,
        *args,
        **kwargs
    ):
        """Initializes the Gaussian source morphology.

        :param source_redshift: Redshift of the source.
        :param cosmo: Astropy cosmology object for angle calculations.
        :param source_size: Size of the source in arcseconds. This
            should be the FWHM of the gaussian. FWHM = 2.3548 * sigma
        :param length_x: Length of the kernel map in x direction in
            arcseconds. Make sure this is larger than the source size.
        :param length_y: Length of the kernel map in y direction in
            arcseconds.
        :param num_pix_x: Number of pixels in x direction.
        :param num_pix_y: Number of pixels in y direction.
        :param center_x: Center of the kernel map in x direction in
            arcseconds.
        :param center_y: Center of the kernel map in y direction in
            arcseconds.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.source_redshift = source_redshift
        self.cosmo = cosmo
        self.source_size = source_size
        self._length_x = length_x
        self._length_y = length_y
        self._num_pix_x = num_pix_x
        self._num_pix_y = num_pix_y
        self.center_x = center_x
        self.center_y = center_y
        self._pixel_scale_x = length_x / num_pix_x
        self._pixel_scale_y = length_y / num_pix_y
        self._pixel_scale = np.sqrt(self._pixel_scale_x * self._pixel_scale_y)
        self._kernel_map = self.get_kernel_map()

    def get_kernel_map(self):
        """Returns the 2D array of the Gaussian kernel map. The kernel map is a
        2D array that represents the morphology of the source. The kernel map
        is used to convolve with the microlensing magnification map. The kernel
        is normalized to 1.

        :return: 2D array of the AGN kernel map.
        """
        xs = np.linspace(
            self.center_x - self.length_x / 2,
            self.center_y + self.length_x / 2,
            self.num_pix_x,
        )
        ys = np.linspace(
            self.center_x - self.length_y / 2,
            self.center_y + self.length_y / 2,
            self.num_pix_y,
        )
        X, Y = np.meshgrid(xs, ys)
        # use normal distribution for the source kernel
        sigma = self.source_size / 2.3548  # FWHM to sigma conversion
        source_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        source_kernel /= np.sum(source_kernel)  # normalize the kernel

        return source_kernel


class AGNSourceMorphology(SourceMorphology):
    """Class for AGN source morphology."""

    def __init__(
        self,
        source_redshift,
        cosmo,
        r_out=1000,
        r_resolution=1000,
        smbh_mass_exp=8.0,
        inclination_angle=0,
        black_hole_spin=0,  # Spin of the black hole
        observer_frame_wavelength_in_nm=600,  # Wavelength in nanometers used to determine black body flux. For the surface flux density of the AccretionDisk at desired wavelength.
        eddington_ratio=0.15,  # Eddington ratio of the accretion disk
        observing_wavelength_band: str = None,
        *args,
        **kwargs
    ):
        """Initializes the AGN source morphology.

        :param source_redshift: Redshift of the source
        :param cosmo: Astropy cosmology object for angle calculations
        :param r_out: Outer radius of the accretion disk in
            gravitational radii. This typically can be chosen as 10^3 to
            10^5 [R_g] Default is 1000.
        :param r_resolution: Resolution of the accretion disk in
            gravitational radii. Default is 1000.
        :param smbh_mass_exp: Exponent of the mass of the supermassive
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
        """
        super().__init__(*args, **kwargs)
        self.source_redshift = source_redshift
        self.cosmo = cosmo
        self.r_out = r_out
        self.r_resolution = r_resolution
        self.smbh_mass_exp = smbh_mass_exp
        self.inclination_angle = inclination_angle
        self.black_hole_spin = black_hole_spin
        self.observer_frame_wavelength_in_nm = observer_frame_wavelength_in_nm
        self.observing_wavelength_band = observing_wavelength_band
        self.eddington_ratio = eddington_ratio
        self.black_hole_mass = 10**smbh_mass_exp

        # -----------------------------------------------------------
        # assign the observer frame wavelength in nm if the band is provided
        # -----------------------------------------------------------
        if self.observing_wavelength_band is not None:
            # Get the mean wavelength of the band
            filter = speclite.filters.load_filter(
                "lsst2023-" + self.observing_wavelength_band
            )
            self.observer_frame_wavelength_in_nm = filter.effective_wavelength.to(
                u.nm
            ).value
            # TODO: In future handle this by integrating the flux map over the band
        # -----------------------------------------------------------

        self._kernel_map = self.get_kernel_map()

        # -----------------------------------------------------------
        # determine the size of the emission map in source plane, in meters
        # -----------------------------------------------------------
        gravitational_radius_of_smbh = calculate_gravitational_radius(
            self.smbh_mass_exp
        )
        kernel_pixel_size_x_metres = (
            2
            * (self.r_out * gravitational_radius_of_smbh)
            / np.size(self.kernel_map, 0)
        )
        kernel_pixel_size_y_metres = (
            2
            * (self.r_out * gravitational_radius_of_smbh)
            / np.size(self.kernel_map, 1)
        )

        self._pixel_scale_x_m = kernel_pixel_size_x_metres.to(
            u.m
        ).value  # convert to meters
        self._pixel_scale_y_m = kernel_pixel_size_y_metres.to(u.m).value

        self._pixel_scale_m = np.sqrt(self._pixel_scale_x_m * self._pixel_scale_y_m)
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # convert pixel scale to arcseconds
        # -----------------------------------------------------------
        self._pixel_scale_x = self.metres_to_arcsecs(
            self._pixel_scale_x_m, self.cosmo, self.source_redshift
        )
        self._pixel_scale_y = self.metres_to_arcsecs(
            self._pixel_scale_y_m, self.cosmo, self.source_redshift
        )
        self._pixel_scale = np.sqrt(self._pixel_scale_x * self._pixel_scale_y)
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # num pixels and length of the kernel map in arcseconds
        # -----------------------------------------------------------
        self._num_pix_x = np.size(self.kernel_map, 0)
        self._num_pix_y = np.size(self.kernel_map, 1)
        self._length_x = self.num_pix_x * self.pixel_scale_x
        self._length_y = self.num_pix_y * self.pixel_scale_y
        # -----------------------------------------------------------

    def get_kernel_map(self):
        """Returns the 2D array of the AGN kernel map. The kernel map is a 2D
        array that represents the morphology of the source. The kernel map is
        used to convolve with the microlensing magnification map. The kernel is
        normalized to 1.

        :return: 2D array of the AGN kernel map.
        """

        # invert redshifts to find locally emitted wavelengths
        redshiftfactor = 1 / (1 + self.source_redshift)
        totalshiftfactor = redshiftfactor  # * self.g_array # we are not using the g_array for now, # TODO: Check with Henry if this is needed?
        rest_frame_wavelength = totalshiftfactor * self.observer_frame_wavelength_in_nm

        accretion_disk_emission_map = calculate_accretion_disk_emission(
            r_out=self.r_out,
            r_resolution=self.r_resolution,
            inclination_angle=self.inclination_angle,
            rest_frame_wavelength_in_nanometers=rest_frame_wavelength,
            black_hole_mass_exponent=self.smbh_mass_exp,
            black_hole_spin=self.black_hole_spin,
            eddington_ratio=self.eddington_ratio,
            return_spectral_radiance_distribution=True,
        )
        accretion_disk_emission_map = np.array(accretion_disk_emission_map)

        # normalize the accretion disk emission map
        normalized_emission_map = accretion_disk_emission_map / np.sum(
            accretion_disk_emission_map
        )

        return normalized_emission_map

    def get_variable_kernel_map(self, *args, **kwargs):
        """Returns the 2D array of the variable AGN kernel map.

        The kernel map is a 2D array that represents the morphology of
        the source. The kernel map is used to convolve with the
        microlensing magnification map. The kernel is normalized to 1.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def get_integrated_kernel_map(self, band):
        """Returns the 2D array of the integrated AGN kernel map for a given
        wavelength range and transmission function.

        The kernel map is a 2D array that represents the morphology of
        the source. The kernel map is used to convolve with the
        microlensing magnification map. The kernel is normalized to 1.
        """
        raise NotImplementedError("This method is not implemented yet.")


class SupernovaeSourceMorphology(SourceMorphology):
    """Class for Supernovae source morphology."""

    def __init__(self, *args, **kwargs):
        """Initializes the Supernovae source morphology."""
        super().__init__(*args, **kwargs)
        raise NotImplementedError(
            "Supernovae source morphology is not implemented yet."
        )
