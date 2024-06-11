from slsim.Deflectors.DeflectorTypes.nfw_hernquist import NFWHernquist
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class TestNFWHernquist(object):
    """
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

    def setup_method(self):
        self.deflector_dict = {
            "halo_mass": 10**13,
            "halo_mass_acc": 0.0,
            "concentration": 10,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "stellar_mass": 10e11,
            "angular_size": 0.001,
            "e1_light": -0.1,
            "e2_light": 0.1,
            "z": 0.5,
            "mag_g": -20,
        }
        self.nfw_hernquist = NFWHernquist(deflector_dict=self.deflector_dict)

    def test_redshift(self):
        z = self.nfw_hernquist.redshift
        assert self.deflector_dict["z"] == z

    def test_halo_properties(self):
        m_halo, c = self.nfw_hernquist.halo_properties
        assert m_halo == self.deflector_dict["halo_mass"]
        assert c == self.deflector_dict["concentration"]

    def test_velocity_dispersion(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        vel_disp = self.nfw_hernquist.velocity_dispersion(cosmo=cosmo)
        npt.assert_almost_equal(vel_disp, 176, decimal=-1)
        assert self.nfw_hernquist.velocity_dispersion(cosmo=cosmo) == vel_disp

    def test_light_model_lenstronomy(self):
        lens_light_model_list, kwargs_lens_light = (
            self.nfw_hernquist.light_model_lenstronomy(band="g")
        )
        assert len(lens_light_model_list) == 1

    def test_mass_model_lenstronomy(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.deflector_dict["z"], z_source=2.0
        )
        lens_mass_model_list, kwargs_lens_mass = (
            self.nfw_hernquist.mass_model_lenstronomy(lens_cosmo=lens_cosmo)
        )
        assert len(lens_mass_model_list) == 2
