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

    def __init__(self, subhalos, cored_profile=False, **deflector_dict):
        """

        :param deflector_dict:  parameters of the cluster halo
        :type deflector_dict: dict
        :param cored_profile: option to add cored density profile
        'PJAFFE' (or 'PJAFFE_ELLIPSE_POTENTIAL') for the main
         deflector halo (default is False)
        :type cored_profile: boolean
        """
        subhalos_list = subhalos
        self._subhalos = [EPLSersic(**subhalo_dict) for subhalo_dict in subhalos_list]
        self._cored_profile = cored_profile
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
        cored = self.cored_profile
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
        if spherical:
            lens_mass_model_list = ["NFW"]
        else:
            lens_mass_model_list = ["NFW_ELLIPSE_CSE"]
            e1_mass, e2_mass = self.mass_ellipticity
            e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
                e1_slsim=e1_mass, e2_slsim=e2_mass
            )
            kwargs_lens_mass[0]["e1"] = e1_mass_lenstronomy
            kwargs_lens_mass[0]["e2"] = e2_mass_lenstronomy

        if cored:
            r_cmin = max(0.5 * 0.2, 0.4 * 0.8)  # r_cmin >= (1/2 pixel scale, 0.4 FWHM)
            r_smax = rs_halo
            r_s = np.random.uniform(r_cmin / 0.3, r_smax)
            r_cmax = 0.3 * r_s  # assuming Rs in NFW could be used for PSEUDOJAFFE
            r_c = np.random.uniform(r_cmin, r_cmax)

            vel_disp = self.velocity_dispersion()
            sigma0 = lens_cosmo.vel_disp_dPIED_sigma0(vel_disp, r_c, r_s)

            kwargs_lens_mass_cored = [
                {
                    "sigma0": sigma0,
                    "Rs": r_s,
                    "Ra": r_c,
                    "center_x": center_lens[0],
                    "center_y": center_lens[1],
                },
            ]

            if spherical:
                lens_mass_model_list += ["PJAFFE"]
            else:
                lens_mass_model_list += ["PJAFFE_ELLIPSE_POTENTIAL"]
                kwargs_lens_mass_cored[0]["e1"] = e1_mass_lenstronomy
                kwargs_lens_mass_cored[0]["e2"] = e2_mass_lenstronomy

            kwargs_lens_mass += kwargs_lens_mass_cored

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

    @property
    def subhalo_redshifts(self):
        """Redshifts of the subhalos for multi-plane LensModel()

        :return: list of subhalo redshifts.
        """
        subhalo_z = []
        for subhalo in self._subhalos:
            subhalo_z.append(float(subhalo.redshift))
        return subhalo_z

    @property
    def cored_profile(self):
        """Boolean flag for cored density profile.

        :return: True for cored, False for cuspy profile.
        """

        return self._cored_profile
