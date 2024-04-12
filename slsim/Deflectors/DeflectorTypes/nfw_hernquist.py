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
        """Velocity dispersion of deflector.

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """

        return self._deflector_dict["vel_disp"]

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

        :return: halo virial mass Mvir [physical M_sol], concentration rvir/rs
        (In SL-hammock code, we now adopt the FOF mass definition, but treat it as the virial mass)
        """
        return self._deflector_dict["halo_mass"], self._deflector_dict["concentration"]
