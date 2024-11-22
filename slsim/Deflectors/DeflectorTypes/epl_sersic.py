from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Deflectors.velocity_dispersion import theta_E_from_vel_disp_epl


class EPLSersic(DeflectorBase):
    """Deflector with an elliptical power-law and a Sersic light model.

    required quantities in dictionary:
    - 'vel_disp': SIS equivalent velocity dispersion of the deflector
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'stellar_mass': stellar mass in physical M_sol
    - 'angular_size': half-light radius of stellar/light profile in radian
    - 'e1_light': eccentricity of light
    - 'e2_light': eccentricity of light
    - 'z': redshift of deflector
    """
    def __init__(self, deflector_dict):
        """

        :param deflector_dict: dictionary of deflector quantities
        """
        super().__init__(deflector_dict=deflector_dict)
        try:
            sis_convention = deflector_dict["sis_convention"]
        except KeyError:
            sis_convention = True
        self._sis_convention = sis_convention

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """

        return self._deflector_dict["vel_disp"]

    @property
    def light_ellipticity(self):
        """Light ellipticity.

        :return: e1_light, e2_light
        """
        e1_light, e2_light = float(self._deflector_dict["e1_light"]), float(
            self._deflector_dict["e2_light"]
        )
        return e1_light, e2_light

    def mass_model_lenstronomy(self, lens_cosmo):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        gamma = self.halo_properties
        if lens_cosmo.z_lens >= lens_cosmo.z_source:
            theta_E = 0.0
        else:
            lens_light_model_list, kwargs_lens_light = self.light_model_lenstronomy()
            theta_E = theta_E_from_vel_disp_epl(vel_disp=float(self.velocity_dispersion(cosmo=lens_cosmo.background.cosmo)),
                                                gamma=gamma,
                                                r_half=self.angular_size_light,
                                                kwargs_light=kwargs_lens_light, light_model_list=lens_light_model_list,
                                                lens_cosmo=lens_cosmo,
                                                kappa_ext=0, sis_convention=self._sis_convention)

        e1_mass, e2_mass = self.mass_ellipticity
        e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
            e1_slsim=e1_mass, e2_slsim=e2_mass
        )
        kwargs_lens_mass = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1_mass_lenstronomy,
                "e2": e2_mass_lenstronomy,
                "center_x": self.deflector_center[0],
                "center_y": self.deflector_center[1],
            }
        ]
        if gamma == 2:
            lens_mass_model_list = ["SIE"]
            kwargs_lens_mass[0].pop("gamma")
        else:
            lens_mass_model_list = ["EPL"]
        return lens_mass_model_list, kwargs_lens_mass

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
        e1_light_lens_lenstronomy, e2_light_lens_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=e1_light_lens, e2_slsim=e2_light_lens
            )
        )
        size_lens_arcsec = self._deflector_dict["angular_size"]
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

    @property
    def halo_properties(self):
        """Mass density logarithmic slope.

        :return: gamma (with =2 is isothermal)
        """
        #if hasattr(self._deflector_dict, "gamma_pl"):
        #    return float(self._deflector_dict["gamma_pl"])
        #else:
        #    # TODO: this can (optionally) be made a function of stellar mass, velocity dispersion etc
        #    return 2
        try:
            return float(self._deflector_dict["gamma_pl"])
        except KeyError:
            return 2
