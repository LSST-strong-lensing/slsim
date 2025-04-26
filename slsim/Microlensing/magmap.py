__author__ = "Paras Sharma"

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

        self.kappa_tot = kappa_tot
        self.shear = shear
        self.kappa_star = kappa_star
        self.theta_star = theta_star
        self.mass_function = mass_function
        self.m_solar = m_solar
        self.m_lower = m_lower
        self.m_upper = m_upper
        self.center_x = center_x
        self.center_y = center_y
        self.half_length_x = half_length_x
        self.half_length_y = half_length_y
        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y

        if self.mass_function is None:
            self.mass_function = "kroupa"
        if self.m_solar is None:
            self.m_solar = 1
        if self.m_lower is None:
            self.m_lower = 0.08
        if self.m_upper is None:
            self.m_upper = 100

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

            self.microlensing_IPM = IPM(
                verbose=1,
                kappa_tot=self.kappa_tot,
                shear=self.shear,
                kappa_star=self.kappa_star,
                smooth_fraction=self.smooth_fraction,
                theta_star=self.theta_star,
                center_y1=self.center_x,
                center_y2=self.center_y,
                half_length_y1=self.half_length_x,
                half_length_y2=self.half_length_y,
                mass_function=self.mass_function,
                m_lower=self.m_lower,
                m_upper=self.m_upper,
                m_solar=self.m_solar,
                num_pixels_y1=self.num_pixels_x,
                num_pixels_y2=self.num_pixels_y,
                approx=True,
                write_maps=False,
                write_parities=False,
                write_histograms=False,
                **kwargs_IPM,
            )

            self.microlensing_IPM.run()
            self.magnifications = (
                self.microlensing_IPM.magnifications
            )  # based on updated IPM class

    def plot_magnification_map(self, ax=None, plot_magnitude=True, **kwargs):
        """Plot the magnification map on the given axis."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if plot_magnitude:
            im = ax.imshow(
                self.magnitudes,
                extent=[
                    (self.center_x - self.half_length_x) / self.theta_star,
                    (self.center_x + self.half_length_x) / self.theta_star,
                    (self.center_y - self.half_length_y) / self.theta_star,
                    (self.center_y + self.half_length_y) / self.theta_star,
                ],
                **kwargs,
            )
        else:
            im = ax.imshow(
                self.magnifications,
                extent=[
                    (self.center_x - self.half_length_x) / self.theta_star,
                    (self.center_x + self.half_length_x) / self.theta_star,
                    (self.center_y - self.half_length_y) / self.theta_star,
                    (self.center_y + self.half_length_y) / self.theta_star,
                ],
                **kwargs,
            )
        ax.set_xlabel("$x / \\theta_★$")
        ax.set_ylabel("$y / \\theta_★$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        if plot_magnitude:
            cbar.set_label("Microlensing $\\Delta m$ (magnitudes)")
        else:
            cbar.set_label("Microlensing magnification")

    @property
    def mu_ave(self):
        return 1 / ((1 - self.kappa_tot) ** 2 - self.shear**2)

    @property
    def stellar_fraction(self):
        return self.kappa_star / self.kappa_tot

    @property
    def smooth_fraction(self):
        return 1 - self.kappa_star / self.kappa_tot

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
        """Returns the pixel scales in (x, y) format."""
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
        """Returns the pixel size in meters."""
        pixel_size_arcsec = self.pixel_size
        pixel_size_meters = (
            cosmo.angular_diameter_distance(source_redshift).to(u.m)
            * pixel_size_arcsec
            * (u.arcsec.to(u.rad))
        )

        return pixel_size_meters.value
