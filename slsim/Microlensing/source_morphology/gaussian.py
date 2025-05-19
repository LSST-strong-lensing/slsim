__author__ = "Paras Sharma"

import numpy as np

from slsim.Microlensing.source_morphology.source_morphology import (
    SourceMorphology,
)


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
