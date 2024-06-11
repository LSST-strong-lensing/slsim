from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from lenstronomy.Util import constants


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
        if lens_cosmo.z_lens >= lens_cosmo.z_source:
            theta_E = 0.0
        else:
            theta_E = lens_cosmo.sis_sigma_v2theta_E(
                float(self.velocity_dispersion(cosmo=lens_cosmo.background.cosmo))
            )
        gamma = self.halo_properties
        e1_mass, e2_mass = self.mass_ellipticity
        kwargs_lens_mass = [
            {
                "theta_E": theta_E,
                "gamma": gamma,
                "e1": e1_mass,
                "e2": e2_mass,
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
        size_lens_arcsec = (
            self._deflector_dict["angular_size"] / constants.arcsec
        )  # convert radian to arc seconds
        lens_light_model_list = ["SERSIC_ELLIPSE"]
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

    @property
    def halo_properties(self):
        """Mass density logarithmic slope.

        :return: gamma (with =2 is isothermal)
        """

        try:
            return float(self._deflector_dict["gamma_pl"])
        except KeyError:
            return 2
