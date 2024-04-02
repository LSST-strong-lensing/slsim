from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from lenstronomy.Util import constants


class NFWHernquist(DeflectorBase):

    """Class of a NFW+Hernquist lens model with a Hernquist light mode"""

    @property
    def velocity_dispersion(self):
        """Velocity dispersion of deflector.

        :return: velocity dispersion [km/s]
        """
        # TODO implement kinematics calculation averaged over half-light radius
        raise NotImplementedError()

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        if band is None:
            mag_lens = 1
        else:
            mag_lens = self.magnitude(band)
        center_lens = self.deflector_center
        e1_light_lens, e2_light_lens = self.light_ellipticity
        size_lens_arcsec = (
                self._deflector_dict["angular_size"] / constants.arcsec
        )  # convert radian to arc seconds

        lens_light_model_list = ["HERNQUIST_ELLIPSE"]
        kwargs_lens_light = [
            {
                "magnitude": mag_lens,
                "Rs": size_lens_arcsec,
                "e1": e1_light_lens,
                "e2": e2_light_lens,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            }
        ]
        return lens_light_model_list, kwargs_lens_light
