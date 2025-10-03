from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_composite_model,
)
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from lenstronomy.Util import constants


class NFWHernquist(DeflectorBase):
    """Class of a NFW+Hernquist lens model with a Hernquist light mode.

    required quantities in dictionary:
    - 'halo_mass': halo mass in physical M_sol
    - 'concentration': halo concentration
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'stellar_mass': stellar mass in physical M_sol
    - 'angular_size': half-light radius of stellar/light profile in radian
    - 'e1_light': eccentricity of light
    - 'e2_light': eccentricity of light
    - 'z': redshift of deflector
    """

    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector. Simplified assumptions on
        anisotropy and averaged over the half-light radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        if ("vel_disp" in self._deflector_dict) and (
            self._deflector_dict["vel_disp"] >= 0
        ):
            return self._deflector_dict["vel_disp"]

        else:
            size_lens_arcsec = self.angular_size_light

            m_halo, c_halo = self.halo_properties
            m_halo_acc = self._deflector_dict["halo_mass_acc"]
            m_halo = max(m_halo, m_halo_acc)
            # convert angular size to physical size. For this, need to convert angular
            # size to radian.
            dd = cosmo.angular_diameter_distance(self.redshift).value
            rs_star = dd * (self.angular_size_light * constants.arcsec)
            vel_disp = vel_disp_composite_model(
                r=size_lens_arcsec,
                m_star=self.stellar_mass,
                rs_star=rs_star,
                m_halo=m_halo,
                c_halo=c_halo,
                cosmo=cosmo,
                z_lens=self.redshift,
            )
            self._deflector_dict["vel_disp"] = vel_disp
            return self._deflector_dict["vel_disp"]

    def mass_model_lenstronomy(self, lens_cosmo, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        if spherical is True:
            lens_mass_model_list = ["NFW", "HERNQUIST"]
        else:
            lens_mass_model_list = ["NFW_ELLIPSE_CSE", "HERNQUIST_ELLIPSE_CSE"]
        e1_light_lens, e2_light_lens = self.light_ellipticity
        e1_light_lens_lenstronomy, e2_light_lens_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=e1_light_lens, e2_slsim=e2_light_lens
            )
        )
        e1_mass, e2_mass = self.mass_ellipticity
        e1_mass_lenstronomy, e2_mass_lenstronomy = ellipticity_slsim_to_lenstronomy(
            e1_slsim=e1_mass, e2_slsim=e2_mass
        )
        rs_phys = lens_cosmo.dd * (self.angular_size_light * constants.arcsec)
        sigma0, rs_light_angle = lens_cosmo.hernquist_phys2angular(
            mass=self.stellar_mass, rs=rs_phys
        )
        # halo mass, concentration, stellar mass
        m_halo, c_halo = self.halo_properties
        rs_halo, alpha_rs = lens_cosmo.nfw_physical2angle(M=m_halo, c=c_halo)
        kwargs_lens_mass = [
            {
                "alpha_Rs": alpha_rs,
                "Rs": rs_halo,
                "center_x": self.deflector_center[0],
                "center_y": self.deflector_center[1],
            },
            {
                "Rs": rs_light_angle,
                "sigma0": sigma0,
                "center_x": self.deflector_center[0],
                "center_y": self.deflector_center[1],
            },
        ]
        if spherical is False:
            kwargs_lens_mass[0]["e1"] = e1_mass_lenstronomy
            kwargs_lens_mass[0]["e2"] = e2_mass_lenstronomy
            kwargs_lens_mass[1]["e1"] = e1_light_lens_lenstronomy
            kwargs_lens_mass[1]["e2"] = e2_light_lens_lenstronomy
        return lens_mass_model_list, kwargs_lens_mass

    def light_model_lenstronomy(self, band=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

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
        e1_light_lens_lenstronomy, e2_light_lens_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=e1_light_lens, e2_slsim=e2_light_lens
            )
        )
        size_lens_arcsec = self.angular_size_light
        lens_light_model_list = ["HERNQUIST_ELLIPSE"]
        kwargs_lens_light = [
            {
                "magnitude": mag_lens,
                "Rs": size_lens_arcsec,
                "e1": e1_light_lens_lenstronomy,
                "e2": e2_light_lens_lenstronomy,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            }
        ]
        return lens_light_model_list, kwargs_lens_light

    @property
    def halo_properties(self):
        """Properties of the NFW halo.

        :return: halo mass M200 [physical M_sol], concentration r200/rs
        """
        return self._deflector_dict["halo_mass"], self._deflector_dict["concentration"]
