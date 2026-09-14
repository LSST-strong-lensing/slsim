from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Deflectors.MassTypes.mass_base import MassBase
from slsim.Deflectors.MassLightConnection.velocity_dispersion import vel_disp_nfw


class NFW(MassBase):
    """Class of a NFW lens model."""

    def __init__(self, light, halo_mass, concentration, e1=0, e2=0, vel_disp=None):
        """

        :param light: light model (used for position of deflector and stellar mass density profile)
        :type light: ~slsim.Sources.source.Source() class instance of Hernquist model or comparable light profile with
         half-light radius that can be interpreted as a Hernquist model
        :param halo_mass: halo mass M200 [physical M_sol]
        :param concentration: halo concentration r200/rs
        :param e1: halo eccentricity component 1
        :param e2: halo eccentricity component 2
        :param vel_disp: velocity dispersion [km/s], optional as pre-computed value.
         ATTENTION: consistency is not checked with mass profile.
        """
        super().__init__(light=light, vel_disp=vel_disp)
        self._halo_mass = halo_mass
        self._concentration = concentration
        self._e1_mass, self._e2_mass = e1, e2

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on
        anisotropy and averaged over the half-light radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        if self._vel_disp is None:
            self._vel_disp = vel_disp_nfw(
                self._halo_mass,
                self._concentration,
                cosmo=cosmo,
                z_lens=self._light.redshift,
            )
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

        if _spherical is True:
            lens_mass_model_list = ["NFW"]
        else:
            lens_mass_model_list = ["NFW_ELLIPSE_CSE"]

        # halo mass, concentration, stellar mass
        rs_halo, alpha_rs = lens_cosmo.nfw_physical2angle(
            M=self._halo_mass, c=self._concentration
        )
        center_x, center_y = self._light.extended_source_position
        kwargs_lens_mass = [
            {
                "alpha_Rs": alpha_rs,
                "Rs": rs_halo,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        if _spherical is False:
            e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1_mass, e2_slsim=self._e2_mass
            )
            kwargs_lens_mass[0]["e1"] = e1_mass_lenstronomy
            kwargs_lens_mass[0]["e2"] = e2_mass_lenstronomy
        return lens_mass_model_list, kwargs_lens_mass

    @property
    def mass_properties(self):
        """Properties of the NFW halo.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return {"halo_mass": self._halo_mass, "concentration": self._concentration}

    @property
    def ellipticity(self):
        """Deflector eccentricities.

        :return: e1, e2
        """
        return self._e1_mass, self._e2_mass
