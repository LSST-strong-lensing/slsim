from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from lenstronomy.Util import constants
from slsim.Deflectors.MassTypes.mass_base import MassBase


class Hernquist(MassBase):
    """Class of a Hernquist lens model from a Hernquist light mode."""

    def __init__(self, light, vel_disp=None):
        """

        :param light: light model (used for position of deflector and stellar mass density profile)
        :type light: ~slsim.Sources.source.Source() class instance of Hernquist model or comparable light profile with
         half-light radius that can be interpreted as a Hernquist model
        :param vel_disp: velocity dispersion [km/s], optional as pre-computed value.
         ATTENTION: consistency is not checked with mass profile.
        """
        super().__init__(light=light, vel_disp=vel_disp)

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on
        anisotropy and averaged over the half-light radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        return self._vel_disp

    def mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        e1_light, e2_light = self._light.ellipticity
        if (spherical is True) or (e1_light == 0 and e2_light == 0):
            _spherical = True
        else:
            _spherical = False
        if _spherical is True:
            lens_mass_model_list = ["HERNQUIST"]
        else:
            lens_mass_model_list = ["HERNQUIST_ELLIPSE_CSE"]

        rs_phys = lens_cosmo.dd * (self._light.angular_size / 1.815 * constants.arcsec)
        sigma0, rs_light_angle = lens_cosmo.hernquist_phys2angular(
            mass=self._light.stellar_mass, rs=rs_phys
        )
        center_x, center_y = self._light.extended_source_position
        kwargs_lens_mass = [
            {
                "Rs": self._light.angular_size / 1.815,
                "sigma0": sigma0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        if _spherical is False:
            e1_light_lenstronomy, e2_light_lenstronomy = (
                ellipticity_slsim_to_lenstronomy(e1_slsim=e1_light, e2_slsim=e2_light)
            )
            kwargs_lens_mass[0]["e1"] = e1_light_lenstronomy
            kwargs_lens_mass[0]["e2"] = e2_light_lenstronomy
        return lens_mass_model_list, kwargs_lens_mass

    @property
    def ellipticity(self):
        """Deflector eccentricities.

        :return: e1, e2
        """
        return self._light.ellipticity
