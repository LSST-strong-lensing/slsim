from abc import ABC, abstractmethod

import numpy as np
from slsim.Util import param_util

_SUPPORTED_DEFLECTORS = ["EPL", "NFW_HERNQUIST"]


class DeflectorBase(ABC):
    """Class of a single deflector with quantities only related to the
    deflector (independent of the source)"""

    def __init__(
        self,
        z,
        vel_disp=None,
        stellar_mass=None,
        angular_size=None,
        center_x=None,
        center_y=None,
        deflector_area=0.01,
        **deflector_dict
    ):
        """

        :param z: redshift
        :param vel_disp: velocity dispersion [km/s]
        :param stellar_mass: stellar mass of deflector [M_sol]
        :param center_x: RA coordinate (relative arcseconds)
        :param center_y: DEC coordinate (relative arcseconds)
        :param angular_size: half light radius of the deflector
         (potentially used to calculate the velocity dispersion of the deflector)
        :param deflector_area: area (in solid angle arcseconds^2) to dither the center of the deflector
         (if center_x or center_y) are not provided
        :param deflector_dict: parameters of the deflector. Assumed to contain:
            'mag_band' for each band, 'e1_light', 'e2_light', 'e1_mass', 'e2_mass',
        :type deflector_dict: dict
        """

        self._vel_disp = vel_disp
        self._z = z
        self._stellar_mass = stellar_mass
        self._deflector_dict = deflector_dict
        self._angular_size = angular_size
        if center_x is None or center_y is None:

            center_x, center_y = param_util.draw_coord_in_circle(
                area=deflector_area, size=1
            )
        self._center_lens = np.array([center_x, center_y])

    def update_center(self, deflector_area):
        """Overwrites the deflector center position.

        :param deflector_area: area (in solid angle arcseconds^2) to
            dither the center of the deflector
        :return:
        """
        center_x, center_y = param_util.draw_coord_in_circle(
            area=deflector_area, size=1
        )
        self._center_lens = np.array([center_x, center_y])

    @property
    def redshift(self):
        """Deflector redshift.

        :return: redshift
        """
        return self._z

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector.

        :param cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        return self._vel_disp

    @property
    def deflector_center(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        return self._center_lens

    @property
    def stellar_mass(self):
        """

        :return: stellar mass of deflector [M_sol]
        """
        return self._stellar_mass

    def magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        band_string = str("mag_" + band)
        return self._deflector_dict[band_string]

    @property
    def light_ellipticity(self):
        """Light ellipticity.

        :return: e1_light, e2_light
        """
        e1_light, e2_light = float(self._deflector_dict["e1_light"]), float(
            self._deflector_dict["e2_light"]
        )
        return e1_light, e2_light

    @property
    def mass_ellipticity(self):
        """Mass ellipticity.

        :return: e1_mass, e2_mass
        """
        e1_mass, e2_mass = float(self._deflector_dict["e1_mass"]), float(
            self._deflector_dict["e2_mass"]
        )
        return e1_mass, e2_mass

    @abstractmethod
    def mass_model_lenstronomy(self, lens_cosmo):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        pass

    @abstractmethod
    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        pass

    @property
    def angular_size_light(self):
        """Angular size of the light component.

        :return: angular size [arcsec]
        """
        return self._angular_size

    @property
    @abstractmethod
    def halo_properties(self):
        """Dictionary of properties of the deflector (mass distribution)

        :return: dictionary of properties
        """
        pass
