from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    theta_E_from_vel_disp_epl,
)
from slsim.Deflectors.MassTypes.mass_base import MassBase


class EPL(MassBase):
    """Deflector with an elliptical power-law and a Sersic light model."""

    def __init__(
        self,
        light,
        theta_E=None,
        vel_disp=None,
        gamma_pl=2,
        e1=0,
        e2=0,
        sis_convention=False,
    ):
        """

        :param light: light model (mostly used for position of deflector)
        :type light: ~slsim.Sources.source.Source() class instance
        :param theta_E: Einstein radius [arcseconds]
         if =None then the Einstein radius is being computed from the velocity dispersion argument
        :param vel_disp: velocity dispersion [km/s].
        :param gamma_pl: logarithmic slope of the mass density profile (2 is isothermal)
        :param sis_convention: if using the SIS convention to normalize the Einstein radius or not
        """
        if theta_E is None and vel_disp is None:
            raise ValueError(
                "Either Einstein radius theta_E or velocity dispersion vel_disp argument need to be "
                "provided for the EPL model."
            )
        super().__init__(light=light, vel_disp=vel_disp)
        self._sis_convention = sis_convention
        self._theta_E = theta_E
        self._gamma_pl = gamma_pl
        self._e1_mass, self._e2_mass = e1, e2

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
            if (
                self._gamma_pl == 2
                or self._sis_convention is True
                or self._light.extended_source_type is None
            ):
                theta_E = lens_cosmo.sis_sigma_v2theta_E(self.velocity_dispersion())
                return theta_E
            else:
                lens_light_model_list, kwargs_lens_light = (
                    self._light.kwargs_extended_light()
                )
                theta_E = theta_E_from_vel_disp_epl(
                    vel_disp=float(self.velocity_dispersion()),
                    gamma=self._gamma_pl,
                    r_half=self._light.angular_size,
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

        e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
            e1_slsim=self._e1_mass, e2_slsim=self._e2_mass
        )
        center_x, center_y = self._light.extended_source_position
        kwargs_lens_mass = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        if (spherical is True) or (self._e1_mass == 0 and self._e2_mass == 0):
            _spherical = True
        else:
            _spherical = False

        if gamma == 2:
            if _spherical is True:
                lens_mass_model_list = ["SIS"]
            else:
                lens_mass_model_list = ["SIE"]
            kwargs_lens_mass[0].pop("gamma")
        else:
            if _spherical is True:
                lens_mass_model_list = ["SPP"]
            else:
                lens_mass_model_list = ["EPL"]
        if _spherical is False:
            kwargs_lens_mass[0]["e1"] = e1_mass_lenstronomy
            kwargs_lens_mass[0]["e2"] = e2_mass_lenstronomy
        return lens_mass_model_list, kwargs_lens_mass

    @property
    def mass_properties(self):
        """Mass density logarithmic slope.

        :return: gamma (with =2 is isothermal), velocity dispersion
            [km/s]
        :rtype: dict
        """
        return {"gamma_pl": self._gamma_pl, "vel_disp": self.velocity_dispersion()}

    @property
    def ellipticity(self):
        """Deflector eccentricities.

        :return: e1, e2
        """
        return self._e1_mass, self._e2_mass
