from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Deflectors.MassLightConnection.velocity_dispersion import vel_disp_nfw
from slsim.Deflectors.DeflectorTypes.epl_sersic import EPLSersic
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
import numpy as np


class NFWCluster(DeflectorBase):
    """Class of a NFW halo lens model with subhalos. Each subhalo is a
    EPLSersic instance with its own mass and light.

    required quantities in dictionary:
    - 'halo_mass': halo mass in physical M_sol
    - 'concentration': halo concentration
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'z': redshift of deflector
    - 'subhalos': list of dictionary with EPLSersic parameters
    """

    def __init__(self, subhalos, **deflector_dict):
        """

        :param deflector_dict:  parameters of the cluster halo
        :type deflector_dict: dict
        """
        subhalos_list = subhalos
        self._subhalos = [EPLSersic(**subhalo_dict) for subhalo_dict in subhalos_list]
        super(NFWCluster, self).__init__(**deflector_dict)

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on
        anisotropy and averaged over the characteristic radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        m_halo, c_halo = self.halo_properties
        return vel_disp_nfw(m_halo, c_halo, cosmo, self.redshift)

    def mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        lens_mass_model_list, kwargs_lens_mass = self._halo_mass_model_lenstronomy(
            lens_cosmo=lens_cosmo, spherical=spherical
        )
        for subhalo in self._subhalos:
            lens_mass_model_list_i, kwargs_lens_mass_i = subhalo.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo, spherical=spherical
            )
            lens_mass_model_list += lens_mass_model_list_i
            kwargs_lens_mass += kwargs_lens_mass_i
        return lens_mass_model_list, kwargs_lens_mass

    def _halo_mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions for the main halo.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        if spherical:
            lens_mass_model_list = ["NFW"]
        else:
            lens_mass_model_list = ["NFW_ELLIPSE_CSE"]

        center_lens = self.deflector_center
        m_halo, c_halo = self.halo_properties
        rs_halo, alpha_rs = lens_cosmo.nfw_physical2angle(M=m_halo, c=c_halo)
        kwargs_lens_mass = [
            {
                "alpha_Rs": alpha_rs,
                "Rs": rs_halo,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            },
        ]
        if not spherical:
            e1_mass, e2_mass = self.mass_ellipticity
            e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
                e1_slsim=e1_mass, e2_slsim=e2_mass
            )
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
        lens_light_model_list, kwargs_lens_light = [], []
        for subhalo in self._subhalos:
            lens_light_model_list_i, kwargs_lens_light_i = (
                subhalo.light_model_lenstronomy(band=band)
            )
            lens_light_model_list += lens_light_model_list_i
            kwargs_lens_light += kwargs_lens_light_i
        return lens_light_model_list, kwargs_lens_light

    @property
    def halo_properties(self):
        """Properties of the NFW halo.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return self._deflector_dict["halo_mass"], self._deflector_dict["concentration"]

    @property
    def stellar_mass(self):
        """

        :return: total stellar mass of deflector [M_sol]
        """
        total_mass = 0
        for subhalo in self._subhalos:
            total_mass += subhalo.stellar_mass
        return total_mass

    def magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: total magnitude of deflector in given band
        """
        total_flux = 0
        for subhalo in self._subhalos:
            mag = subhalo.magnitude(band)
            total_flux += 10 ** (-0.4 * mag)
        return -2.5 * np.log10(total_flux)
