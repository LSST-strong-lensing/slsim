from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Deflectors.MassLightConnection.velocity_dispersion import theta_E_from_vel_disp_epl


class EPLSersic(DeflectorBase):
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

    def __init__(self, deflector_dict, sis_convention=True):
        """

        :param deflector_dict: dictionary of deflector quantities
        :param sis_convention: if using the SIS convention to normalize the Einstein radius or not
        """
        super().__init__(deflector_dict=deflector_dict)

        self._sis_convention = sis_convention

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. If velocity dispersion is not
        provided in the deflector dict, None will be returned. Then,
        _einstein_radius() function expects value of einstein radius in the
        deflector dict which will be used in mass_model_lenstronomy() function.

        :param cosmo: cosmology. This parameter is not used in this
            function. We use it as a dummy input for consistency with
            other deflector types. default is None.
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        if "vel_disp" in self._deflector_dict.keys():
            vel_disp = self._deflector_dict["vel_disp"]
        else:
            vel_disp = None
        return vel_disp

    def _einstein_radius(self, lens_cosmo=None):
        """Einstein radius of the deflector.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: Einstein radius of the deflector
        """
        if "theta_E" in self._deflector_dict.keys():
            theta_E = self._deflector_dict[
                "theta_E"
            ]  # given einstein radius should be in arcsec.
        else:
            lens_light_model_list, kwargs_lens_light = self.light_model_lenstronomy()
            theta_E = theta_E_from_vel_disp_epl(
                vel_disp=float(self.velocity_dispersion()),
                gamma=self.halo_properties,
                r_half=self.angular_size_light,
                kwargs_light=kwargs_lens_light,
                light_model_list=lens_light_model_list,
                lens_cosmo=lens_cosmo,
                kappa_ext=0,
                sis_convention=self._sis_convention,
            )
        return theta_E

    @property
    def light_ellipticity(self):
        """Light ellipticity.

        :return: e1_light, e2_light
        """
        e1_light, e2_light = float(self._deflector_dict["e1_light"]), float(
            self._deflector_dict["e2_light"]
        )
        return e1_light, e2_light

    def mass_model_lenstronomy(self, lens_cosmo=None, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        gamma = self.halo_properties
        if lens_cosmo.z_lens >= lens_cosmo.z_source:
            theta_E = 0.0
        else:
            theta_E = self._einstein_radius(lens_cosmo=lens_cosmo)

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
            if spherical is True:
                lens_mass_model_list = ["SIS"]
            else:
                lens_mass_model_list = ["SIE"]
            kwargs_lens_mass[0].pop("gamma")
        else:
            if spherical is True:
                lens_mass_model_list = ["SPP"]
            else:
                lens_mass_model_list = ["EPL"]
        if spherical is False:
            kwargs_lens_mass[0]["e1"] = e1_mass_lenstronomy
            kwargs_lens_mass[0]["e2"] = e2_mass_lenstronomy
        return lens_mass_model_list, kwargs_lens_mass

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
        try:
            return float(self._deflector_dict["gamma_pl"])
        except KeyError:
            # TODO: this can (optionally) be made a function of stellar mass, velocity dispersion etc
            return 2
