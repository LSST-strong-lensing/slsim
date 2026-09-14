from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_composite_model,
)
from slsim.Deflectors.MassTypes.nfw import NFW
from slsim.Deflectors.MassTypes.hernquist import Hernquist
from lenstronomy.Util import constants
from slsim.Deflectors.MassTypes.mass_base import MassBase


class NFWHernquist(MassBase):
    """Class of a NFW+Hernquist lens model with a Hernquist light mode."""

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
        self._nfw = NFW(
            light=light, halo_mass=halo_mass, concentration=concentration, e1=e1, e2=e2
        )
        self._hernquist = Hernquist(light=light)
        self.num_mass_models = 2

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on
        anisotropy and averaged over the half-light radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        if self._vel_disp is not None:
            if self._vel_disp >= 0:
                return self._vel_disp

        size_lens_arcsec = self._light.angular_size
        # convert angular size to physical size. For this, need to convert angular
        # size to radian.
        dd = cosmo.angular_diameter_distance(self._light.redshift).value
        rs_star = dd * (size_lens_arcsec * constants.arcsec)
        vel_disp = vel_disp_composite_model(
            r=size_lens_arcsec,
            m_star=self._light.stellar_mass,
            rs_star=rs_star,
            m_halo=self._nfw._halo_mass,
            c_halo=self._nfw._concentration,
            cosmo=cosmo,
            z_lens=self._light.redshift,
        )
        self._vel_disp = vel_disp
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
        lens_mass_model_list_nfw, kwargs_lens_mass_nfw = (
            self._nfw.mass_model_lenstronomy(lens_cosmo=lens_cosmo, spherical=spherical)
        )
        lens_mass_model_list_her, kwargs_lens_mass_her = (
            self._hernquist.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo, spherical=spherical
            )
        )

        return (
            lens_mass_model_list_nfw + lens_mass_model_list_her,
            kwargs_lens_mass_nfw + kwargs_lens_mass_her,
        )

    @property
    def mass_properties(self):
        """Properties of the NFW halo.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return self._nfw.mass_properties

    @property
    def ellipticity(self):
        """Deflector eccentricities.

        :return: e1, e2
        """
        return self._nfw.ellipticity
