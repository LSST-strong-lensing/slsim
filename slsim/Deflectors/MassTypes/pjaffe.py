from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Deflectors.MassTypes.mass_base import MassBase


class PJAFFE(MassBase):
    """Class of a pseudo-Jaffe lens model ."""

    def __init__(
        self,
        light,
        r_s,
        r_a,
        vel_disp,
        e1=0,
        e2=0,
    ):
        """

        :param light: light model (used for position of deflector and stellar mass density profile)
        :type light: ~slsim.Sources.source.Source() class instance of Hernquist model or comparable light profile with
         half-light radius that can be interpreted as a Hernquist model
        :param r_s: core radius of pseudo Jaffe profile [arcsec]
        :param r_a: truncation or tidal radius [arcsec] with r_a > r_s
        :param e1: halo eccentricity component 1
        :param e2: halo eccentricity component 2
        :param vel_disp: velocity dispersion [km/s]
         ATTENTION: consistency is not checked with mass profile.
        """
        super().__init__(light=light, vel_disp=vel_disp)
        self._r_s = r_s
        self._r_a = r_a
        self._e1_mass, self._e2_mass = e1, e2

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
        if (spherical is True) or (self._e1_mass == 0 and self._e2_mass == 0):
            _spherical = True
        else:
            _spherical = False

        vel_disp = self.velocity_dispersion()
        sigma0 = lens_cosmo.vel_disp_dPIED_sigma0(vel_disp, Ra=self._r_a, Rs=self._r_s)
        center_x, center_y = self._light.extended_source_position
        kwargs_lens = [
            {
                "sigma0": sigma0,
                "Rs": self._r_s,
                "Ra": self._r_a,
                "center_x": center_x,
                "center_y": center_y,
            },
        ]

        if _spherical:
            lens_mass_model_list = ["PJAFFE"]
        else:
            lens_mass_model_list = ["PJAFFE_ELLIPSE_POTENTIAL"]
            e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1_mass, e2_slsim=self._e2_mass
            )
            kwargs_lens[0]["e1"] = e1_mass_lenstronomy
            kwargs_lens[0]["e2"] = e2_mass_lenstronomy
        return lens_mass_model_list, kwargs_lens

    @property
    def mass_properties(self):
        """Properties of the PJAFFE profile.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return {"r_s": self._r_s, "r_a": self._r_a, "vel_disp": self._vel_disp}

    @property
    def ellipticity(self):
        """Deflector eccentricities.

        :return: e1, e2
        """
        return self._e1_mass, self._e2_mass
