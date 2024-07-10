import pytest

from slsim.Deflectors.DeflectorTypes.epl_sersic import EPLSersic
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class TestEPLSersic(object):
    """
    required quantities in dictionary:
    - 'velocity_dispersion': SIS equivalent velocity dispersion of the deflector
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'angular_size': half-light radius of stellar/light profile in radian
    - 'e1_light': eccentricity of light
    - 'e2_light': eccentricity of light
    - 'z': redshift of deflector
    """

    def setup_method(self):
        self.deflector_dict = {
            "vel_disp": 200,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "angular_size": 0.001,
            "n_sersic": 1,
            "e1_light": -0.1,
            "e2_light": 0.1,
            "z": 0.5,
        }
        self.epl_sersic = EPLSersic(deflector_dict=self.deflector_dict)

    def test_redshift(self):
        z = self.epl_sersic.redshift
        assert self.deflector_dict["z"] == z

    def test_velocity_dispersion(self):
        vel_disp = self.epl_sersic.velocity_dispersion()
        assert vel_disp == self.deflector_dict["vel_disp"]

    def test_mass_model_lenstronomy_sie(self):
        # Should yeld SIE model as gamma = 2
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.epl_sersic.redshift, z_source=2.0
        )
        lens_mass_model_list, kwargs_lens_mass = self.epl_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo
        )
        assert len(lens_mass_model_list) == 1
        assert lens_mass_model_list[0] == "SIE"

    def test_mass_model_no_lensing(self):
        # case when z_source < z_lens
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.epl_sersic.redshift, z_source=0.2
        )
        lens_mass_model_list, kwargs_lens_mass = self.epl_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo
        )
        assert kwargs_lens_mass[0]["theta_E"] == 0.0

    def test_halo_porperties(self):
        gamma = self.epl_sersic.halo_properties
        assert gamma == 2.0


@pytest.fixture
def gamma_epl_sersic_instance():
    deflector_dict = {
        "vel_disp": 200,
        "gamma_pl": 1.9,
        "e1_mass": 0.1,
        "e2_mass": -0.1,
        "angular_size": 0.001,
        "n_sersic": 1,
        "e1_light": -0.1,
        "e2_light": 0.1,
        "z": 0.5,
    }
    return EPLSersic(deflector_dict=deflector_dict)


def test_mass_model_lenstronomy_gamma(gamma_epl_sersic_instance):
    # case when gamma != 2
    gamma_epl_sersic = gamma_epl_sersic_instance
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lens_cosmo = LensCosmo(cosmo=cosmo, z_lens=gamma_epl_sersic.redshift, z_source=2.0)
    lens_mass_model_list, kwargs_lens_mass = gamma_epl_sersic.mass_model_lenstronomy(
        lens_cosmo=lens_cosmo
    )
    assert len(lens_mass_model_list) == 1
    assert lens_mass_model_list[0] == "EPL"


if __name__ == "__main__":
    pytest.main()
