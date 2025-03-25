__author__ = "Paras Sharma"

import sys
import numpy as np

# Credits: Luke's Microlensing code - https://github.com/weisluke/microlensing
from slsim.Microlensing import LUKES_MICROLENSING_PATH

sys.path.append(LUKES_MICROLENSING_PATH)
from microlensing.IPM.ipm import (
    IPM,
)  # Inverse Polygon Mapping class to generate magnification maps


class MagnificationMap(object):
    """Class to generate magnification maps based on the kappa_tot, shear,
    kappa_star, etc."""

    def __init__(
        self,
        kappa_tot: float = None,
        shear: float = None,
        kappa_star: float = None,
        theta_star: float = None,
        mass_function: str = None,
        m_solar: float = None,
        m_lower: float = None,
        m_upper: float = None,
        light_loss: float = None,
        rectangular: bool = None,
        center_x: float = None,
        center_y: float = None,
        half_length_x: float = None,
        half_length_y: float = None,
        num_pixels_x: int = None,
        num_pixels_y: int = None,
        num_rays_y: int = None,
        random_seed: int = None,
    ):
        """
        :param kappa_tot: total convergence
        :param shear: shear
        :param kappa_star: convergence in point mass lenses/stars
        :param theta_star: Einstein radius of a unit mass point lens in arbitrary units. Default is 1.
        :param mass_function: mass function to use for the point mass lenses. Options are: equal, uniform, salpeter, kroupa, and optical_depth. Default is kroupa.
        :param m_solar: mass of the sun in arbitrary units. Default is 1.
        :param m_lower: lower mass limit for the mass function in solar masses. Default is 0.08.
        :param m_upper: upper mass limit for the mass function in solar masses. Default is 100.
        :param light_loss: fraction of light lost due to microlensing (0 <= light_loss <= 1)
        :param rectangular: whether the map is rectangular or not
        :param center_x: x coordinate of the center of the magnification map
        :param center_y: y coordinate of the center of the magnification map
        :param half_length_x: x extent of the half-length of the magnification map
        :param half_length_y: y extent of the half_length of the magnification map
        :param num_pixels_x: number of pixels for the x axis
        :param num_pixels_y: number of pixels for the y axis
        """

        self.kappa_tot = kappa_tot
        self.shear = shear
        self.kappa_star = kappa_star
        self.theta_star = theta_star
        self.mass_function = mass_function
        self.m_solar = m_solar
        self.m_lower = m_lower
        self.m_upper = m_upper
        self.light_loss = light_loss
        self.center_x = center_x
        self.center_y = center_y
        self.half_length_x = half_length_x
        self.half_length_y = half_length_y
        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y
        self.num_rays_y = num_rays_y
        self.random_seed = random_seed

        if self.mass_function is None:
            self.mass_function = "kroupa"
        if self.m_solar is None:
            self.m_solar = 1
        if self.m_lower is None:
            self.m_lower = 0.08
        if self.m_upper is None:
            self.m_upper = 100

        self.microlensing_IPM = IPM(
            verbose=1,
            kappa_tot=self.kappa_tot,
            shear=self.shear,
            kappa_star=self.kappa_star,
            smooth_fraction=self.smooth_fraction,
            theta_star=self.theta_star,
            light_loss=self.light_loss,
            rectangular=rectangular,
            approx=True,
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
            num_rays_y=self.num_rays_y,
            random_seed=self.random_seed,
            write_maps=False,
            write_parities=False,
            write_histograms=False,
        )

        self.generate_magnification_map()
        self.magnifications = self.microlensing_MagMap.magnifications

    def generate_magnification_map(self):
        """Generate the magnification map based on the parameters provided."""
        self.microlensing_MagMap = self.microlensing_IPM.run()

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
            self.magnifications.shape[0],
        )  # axis 0 of array is y2 axis (IPM convention)

    @property
    def pixel_scales(self):
        """Returns the pixel scales in (x, y) format."""
        return (
            2 * self.half_length[0] / self.num_pixels[0],
            2 * self.half_length[1] / self.num_pixels[1],
        )

    @property
    def magnitudes(self):
        """Returns the magnitudes of the magnification map normalized by the
        average magnification."""
        return -2.5 * np.log10(self.magnifications / np.abs(self.mu_ave))
