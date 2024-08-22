from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Deflectors.velocity_dispersion import vel_disp_nfw
from slsim.Deflectors.DeflectorTypes.epl_sersic import EPLSersic
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy


class NFWCluster(DeflectorBase):
    """Class of a NFW halo lens model with subhalos. Each subhalo is a EPLSersic
    instance with its own mass and light.

    required quantities in dictionary:
    - 'halo_mass': halo mass in physical M_sol
    - 'concentration': halo concentration
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'z': redshift of deflector
    - 'subhalos': list of dictionary with subhalo parameters
    """

    def __init__(self, deflector_dict):
        """

        :param deflector_dict:  parameters of the cluster halo
        :type deflector_dict: dict
        """
        subhalos_list = deflector_dict.pop("subhalos")
        self._subhalos = [EPLSersic(subhalo_dict) for subhalo_dict in subhalos_list]
        super(NFWCluster, self).__init__(deflector_dict)

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on anisotropy and
        averaged over the characteristic radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        m_halo, c_halo = self.halo_properties
        return vel_disp_nfw(m_halo, c_halo, cosmo, self.redshift)

    def mass_model_lenstronomy(self, lens_cosmo):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        lens_mass_model_list, kwargs_lens_mass = self._halo_mass_model_lenstronomy(
            lens_cosmo=lens_cosmo
        )
        for subhalo in self._subhalos:
            lens_mass_model_list_i, kwargs_lens_mass_i = subhalo.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo
            )
            lens_mass_model_list += lens_mass_model_list_i
            kwargs_lens_mass += kwargs_lens_mass_i
        return lens_mass_model_list, kwargs_lens_mass

    def _halo_mass_model_lenstronomy(self, lens_cosmo):
        """Returns lens model instance and parameters in lenstronomy conventions for the
        main halo.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        lens_mass_model_list = ["NFW_ELLIPSE_CSE"]
        e1_mass, e2_mass = self.mass_ellipticity
        e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
            e1_slsim=e1_mass, e2_slsim=e2_mass
        )
        center_lens = self.deflector_center
        m_halo, c_halo = self.halo_properties
        rs_halo, alpha_rs = lens_cosmo.nfw_physical2angle(M=m_halo, c=c_halo)
        kwargs_lens_mass = [
            {
                "alpha_Rs": alpha_rs,
                "Rs": rs_halo,
                "e1": e1_mass_lenstronomy,
                "e2": e2_mass_lenstronomy,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            },
        ]
        return lens_mass_model_list, kwargs_lens_mass

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy conventions.

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
