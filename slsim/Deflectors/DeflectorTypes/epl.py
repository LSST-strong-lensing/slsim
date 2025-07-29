from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    theta_E_from_vel_disp_epl,
)


class EPL(DeflectorBase):
    """Deflector with an elliptical power-law and a Sersic light model.

    required quantities in dictionary:
    - 'vel_disp': SIS equivalent velocity dispersion of the deflector
    - 'gamma_pl': power-law slope
    - 'e1_mass': eccentricity of EPL profile
    - 'e2_mass': eccentricity of EPL profile
    - 'stellar_mass': stellar mass in physical M_sol
    - 'angular_size': half-light radius of stellar/light profile in radian
    - 'z': redshift of deflector
    """

    # TODO: add center_x center_y to documentation

    def __init__(self, sis_convention=True, theta_E=None, gamma_pl=2, **deflector_dict):
        """

        :param deflector_dict: dictionary of deflector quantities
        :param theta_E: Einstein radius [arcseconds]
         if =None then the Einstein radius is being computed from the velocity dispersion argument
        :param gamma_pl: logarithmic slope of the mass density profile (2 is isothermal)
        :param sis_convention: if using the SIS convention to normalize the Einstein radius or not
        """
        super().__init__(**deflector_dict)
        self._sis_convention = sis_convention
        self._theta_E = theta_E
        self._gamma_pl = gamma_pl

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
        return self._vel_disp

    def _einstein_radius(self, lens_cosmo=None):
        """Einstein radius of the deflector.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: Einstein radius of the deflector
        """
        if self._theta_E is None:
            lens_light_model_list, kwargs_lens_light = self.light_model_lenstronomy()
            theta_E = theta_E_from_vel_disp_epl(
                vel_disp=float(self.velocity_dispersion()),
                gamma=self._gamma_pl,
                r_half=self.angular_size_light,
                kwargs_light=kwargs_lens_light,
                light_model_list=lens_light_model_list,
                lens_cosmo=lens_cosmo,
                kappa_ext=0,
                sis_convention=self._sis_convention,
            )
            return theta_E
        return self._theta_E

    def mass_model_lenstronomy(self, lens_cosmo=None, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        gamma = self._gamma_pl
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
        lens_light_model_list = []
        kwargs_lens_light = []
        return lens_light_model_list, kwargs_lens_light

    @property
    def halo_properties(self):
        """Mass density logarithmic slope.

        :return: gamma (with =2 is isothermal)
        """
        return {"gamma_pl": self._gamma_pl}
