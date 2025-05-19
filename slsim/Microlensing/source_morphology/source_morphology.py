__author__ = "Paras Sharma"

import numpy as np

from astropy import units as u


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
        return self._num_pix_x

    @property
    def num_pix_y(self):
        """Returns the number of pixels in y direction."""
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
