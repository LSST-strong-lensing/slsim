from slsim.Deflectors.DeflectorTypes.epl import EPL
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy


class EPLSersic(EPL):
    """Deflector with an elliptical power-law and a Sersic light model.

    required quantities in dictionary:
    - 'vel_disp': SIS equivalent velocity dispersion of the deflector
    - 'e1_mass': eccentricity of EPL profile
    - 'e2_mass': eccentricity of EPL profile
    - 'stellar_mass': stellar mass in physical M_sol
    - 'angular_size': half-light radius of stellar/light profile in radian
    - 'n_sersic': Sersic index of deflector light
    - 'e1_light': eccentricity of light
    - 'e2_light': eccentricity of light
    - 'z': redshift of deflector
    """

    # TODO: add center_x center_y to documentation

    def __init__(self, **deflector_dict):
        """

        :param deflector_dict: dictionary of deflector quantities
        :param sis_convention: if using the SIS convention to normalize the Einstein radius or not
        """
        super().__init__(**deflector_dict)

    @property
    def light_ellipticity(self):
        """Light ellipticity.

        :return: e1_light, e2_light
        """
        e1_light, e2_light = float(self._deflector_dict["e1_light"]), float(
            self._deflector_dict["e2_light"]
        )
        return e1_light, e2_light

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

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
        e1_light_lens_lenstronomy, e2_light_lens_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=e1_light_lens, e2_slsim=e2_light_lens
            )
        )
        size_lens_arcsec = self.angular_size_light
        lens_light_model_list = ["SERSIC_ELLIPSE"]
        kwargs_lens_light = [
            {
                "magnitude": mag_lens,
                "R_sersic": size_lens_arcsec,
                "n_sersic": float(self._deflector_dict["n_sersic"]),
                "e1": e1_light_lens_lenstronomy,
                "e2": e2_light_lens_lenstronomy,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            }
        ]
        return lens_light_model_list, kwargs_lens_light
