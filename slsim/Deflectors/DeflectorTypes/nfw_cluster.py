from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Deflectors.velocity_dispersion import vel_disp_nfw_aperture
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class NFWCluster(DeflectorBase):
    """Class of a NFW halo lens model with subhalos. Each subhalo is a Deflector
    instance with its own mass and light.

    required quantities in dictionary:
    - 'halo_mass': halo mass in physical M_sol
    - 'concentration': halo concentration
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'z': redshift of deflector
    """

    def __init__(self, deflector_dict, subhalos_list):
        """

        :param deflector_dict:  parameters of the cluster halo
        :type deflector_dict: dict
        :param subhalos_list: list with Deflector instances as cluster subhalos
        :type subhalos_list: list[Deflector]
        """
        super(NFWCluster, self).__init__(deflector_dict)
        self._subhalos = subhalos_list

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on anisotropy and
        averaged over the characteristic radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        # convert radian to arc seconds
        lens_cosmo = LensCosmo(z_lens=self.redshift, z_source=10, cosmo=cosmo)

        m_halo, c_halo = self.halo_properties
        rs_arcsec, _ = lens_cosmo.nfw_physical2angle(m_halo, c_halo)
        vel_disp = vel_disp_nfw_aperture(
            r=rs_arcsec,
            m_halo=m_halo,
            c_halo=c_halo,
            cosmo=cosmo,
            z_lens=self.redshift,
        )
        return vel_disp

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
        center_lens = self.deflector_center
        m_halo, c_halo = self.halo_properties
        rs_halo, alpha_rs = lens_cosmo.nfw_physical2angle(M=m_halo, c=c_halo)
        kwargs_lens_mass = [
            {
                "alpha_Rs": alpha_rs,
                "Rs": rs_halo,
                "e1": e1_mass,
                "e2": e2_mass,
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
