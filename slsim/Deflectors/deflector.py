import numpy as np
from lenstronomy.Util import constants


class Deflector(object):
    """Class of a single deflector with quantities only related to the deflector
    (independent of the source)"""

    def __init__(self, deflector_dict):
        """

        :param deflector_dict: parameters of the deflector
        :type deflector_dict: dict
        """
        self._deflector_dict = deflector_dict

    @property
    def redshift(self):
        """Deflector redshift.

        :return: redshift
        """
        return self._deflector_dict["z"]

    @property
    def velocity_dispersion(self):
        """Velocity dispersion of deflector.

        :return: velocity dispersion [km/s]
        """
        return self._deflector_dict["vel_disp"]

    @property
    def deflector_center(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        if not hasattr(self, "_center_lens"):
            center_x_lens, center_y_lens = np.random.normal(
                loc=0, scale=0.1
            ), np.random.normal(loc=0, scale=0.1)
            self._center_lens = np.array([center_x_lens, center_y_lens])
        return self._center_lens

    @property
    def stellar_mass(self):
        """

        :return: stellar mass of deflector
        """
        return self._deflector_dict["stellar_mass"]

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

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        lens_light_model_list = ["SERSIC_ELLIPSE"]
        center_lens = self.deflector_center
        e1_light_lens, e2_light_lens = self.light_ellipticity
        size_lens_arcsec = (
            self._deflector_dict["angular_size"] / constants.arcsec
        )  # convert radian to arc seconds

        if band is None:
            mag_lens = 1
        else:
            mag_lens = self.magnitude(band)
        kwargs_lens_light = [
            {
                "magnitude": mag_lens,
                "R_sersic": size_lens_arcsec,
                "n_sersic": float(self._deflector_dict["n_sersic"]),
                "e1": e1_light_lens,
                "e2": e2_light_lens,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            }
        ]
        return lens_light_model_list, kwargs_lens_light
