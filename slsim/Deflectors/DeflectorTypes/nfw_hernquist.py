from slsim.Deflectors.DeflectorTypes.deflector_base import DeflectorBase
from slsim.Deflectors.velocity_dispersion import vel_disp_composite_model
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
        """Velocity dispersion of deflector. Simplified assumptions on anisotropy and
        averaged over the half-light radius.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        # convert radian to arc seconds
        size_lens_arcsec = self.angular_size_light / constants.arcsec

        m_halo, c_halo = self.halo_properties
        # convert angular size to physical size
        dd = cosmo.angular_diameter_distance(self.redshift).value
        rs_star = dd * self.angular_size_light
        vel_disp = vel_disp_composite_model(
            r=size_lens_arcsec,
            m_star=self.stellar_mass,
            rs_star=rs_star,
            m_halo=m_halo,
            c_halo=c_halo,
            cosmo=cosmo,
            z_lens=self.redshift,
        )
        return vel_disp

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

        lens_light_model_list = ["HERNQUIST_ELLIPSE"]
        kwargs_lens_light = [
            {
                "magnitude": mag_lens,
                "Rs": size_lens_arcsec,
                "e1": e1_light_lens,
                "e2": e2_light_lens,
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
