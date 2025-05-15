__author__ = "Paras Sharma"

import numpy as np
from astropy import units as u


class MagnificationMap(object):
    """Class to generate magnification maps based on the kappa_tot, shear,
    kappa_star, etc."""

    def __init__(
        self,
        magnifications_array: np.ndarray = None,
        kappa_tot: float = None,
        shear: float = None,
        kappa_star: float = None,
        theta_star: float = None,
        mass_function: str = None,
        m_solar: float = None,
        m_lower: float = None,
        m_upper: float = None,
        center_x: float = None,
        center_y: float = None,
        half_length_x: float = None,
        half_length_y: float = None,
        num_pixels_x: int = None,
        num_pixels_y: int = None,
        kwargs_IPM: dict = {},
    ):
        """
        :param magnifications_array: array of magnifications to use. If None, a new
            magnification map will be generated based on the parameters
            provided.
        :param kappa_tot: total convergence
        :param shear: shear
        :param kappa_star: convergence in point mass lenses/stars.
        :param theta_star: Einstein radius of a unit mass point lens in arcsec units. Default is 1.
        :param mass_function: mass function to use for the point mass lenses. Options are: equal, uniform, salpeter, kroupa, and optical_depth. Default is kroupa.
        :param m_solar: solar mass in arbitrary units. Default is 1.
        :param m_lower: lower mass limit for the mass function in solar masses. Default is 0.08.
        :param m_upper: upper mass limit for the mass function in solar masses. Default is 100.
        :param center_x: x coordinate of the center of the magnification map in arcsec units. Default is 0.
        :param center_y: y coordinate of the center of the magnification map in arcsec units. Default is 0.
        :param half_length_x: x extent of the half-length of the magnification map in arcsec units.
        :param half_length_y: y extent of the half_length of the magnification map in arcsec units.
        :param num_pixels_x: number of pixels for the x axis
        :param num_pixels_y: number of pixels for the y axis
        :param kwargs_IPM: additional keyword arguments to pass to the IPM class.
        """

        # Private attributes
        self._kappa_tot = kappa_tot
        self._shear = shear
        self._kappa_star = kappa_star
        self._mass_function = mass_function
        self._m_solar = m_solar
        self._m_lower = m_lower
        self._m_upper = m_upper

        # Public attributes
        self.center_x = center_x
        self.center_y = center_y
        self.theta_star = theta_star
        self.half_length_x = half_length_x
        self.half_length_y = half_length_y
        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y

        if self._mass_function is None:
            self._mass_function = "kroupa"
        if self._m_solar is None:
            self._m_solar = 1
        if self._m_lower is None:
            self._m_lower = 0.08
        if self._m_upper is None:
            self._m_upper = 100

        if magnifications_array is not None:
            self.magnifications = magnifications_array  # TODO: make it so that the magnification map is not generated again, is stored in cache!
        else:
            try:
                # Credits: Luke's Microlensing code - https://github.com/weisluke/microlensing
                from microlensing.IPM.ipm import (
                    IPM,
                )  # Inverse Polygon Mapping class to generate magnification maps
            except ImportError:
                raise ImportError(
                    "The microlensing package is not installed. Please install it using 'pip install microlensing'."
                    "And make sure you are on a GPU that supports CUDA."
                )

            self._microlensing_IPM = IPM(
                kappa_tot=self._kappa_tot,
                shear=self._shear,
                kappa_star=self._kappa_star,
                smooth_fraction=self.smooth_fraction,
                theta_star=self.theta_star,
                center_y1=self.center_x,
                center_y2=self.center_y,
                half_length_y1=self.half_length_x,
                half_length_y2=self.half_length_y,
                mass_function=self._mass_function,
                m_lower=self._m_lower,
                m_upper=self._m_upper,
                m_solar=self._m_solar,
                num_pixels_y1=self.num_pixels_x,
                num_pixels_y2=self.num_pixels_y,
                **kwargs_IPM,
            )

            print("Generating magnification map ...")
            self._microlensing_IPM.run()
            print("Done generating magnification map.")
            self.magnifications = (
                self._microlensing_IPM.magnifications
            )  # based on updated IPM class

    @property
    def mu_ave(self):
        """Returns the average (macro) magnification of the magnification
        map."""
        return 1 / ((1 - self._kappa_tot) ** 2 - self._shear**2)

    @property
    def stellar_fraction(self):
        """Returns the convergence fraction of that is due to stars/compact
        objects."""
        return self._kappa_star / self._kappa_tot

    @property
    def smooth_fraction(self):
        """Returns the convergence fraction of that is due to smooth matter."""
        return 1 - self._kappa_star / self._kappa_tot

    @property
    def num_pixels(self):
        """Returns the number of pixels in the magnification map in (x, y)
        format."""
        return (
            self.magnifications.shape[1],  # axis 1 of array is y1 axis (IPM convention)
            self.magnifications.shape[0],  # axis 0 of array is y2 axis (IPM convention)
        )

    @property
    def pixel_scales(self):
        """Returns the pixel scales in (x, y) format.

        The units are arcseconds.
        """
        return (
            2 * self.half_length_x / self.num_pixels_x,
            2 * self.half_length_y / self.num_pixels_y,
        )

    @property
    def pixel_size(self):
        """Returns the pixel size in arcseconds."""
        return np.sqrt(self.pixel_scales[0] * self.pixel_scales[1])

    @property
    def magnitudes(self):
        """Returns the magnitudes of the magnification map normalized by the
        average magnification."""
        return -2.5 * np.log10(self.magnifications / np.abs(self.mu_ave))

    def get_pixel_size_meters(self, source_redshift, cosmo):
        """Returns the pixel size in meters.

        :param source_redshift: redshift of the source.
        :param cosmo: astropy.cosmology instance.
        :return: pixel size in meters.
        """
        pixel_size_arcsec = self.pixel_size
        pixel_size_meters = (
            cosmo.angular_diameter_distance(source_redshift).to(u.m)
            * pixel_size_arcsec
            * (u.arcsec.to(u.rad))
        )

        return pixel_size_meters.value
