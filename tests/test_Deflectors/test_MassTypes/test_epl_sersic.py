import pytest
from slsim.Deflectors.MassTypes.epl import EPL
from slsim.Sources.source import Source
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

        self.kwargs_source = {
            "extended_source_type": "single_sersic",
            "angular_size": 0.1,
            "n_sersic": 1,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0,
            "center_y": 0,
            "z": 0.5,
        }

        source = Source(**self.kwargs_source)

        self.deflector_dict = {
            "vel_disp": 200,
            "e1": 0.1,
            "e2": -0.1,
            "gamma_pl": 2,
        }
        # gamma_pl not given, hence using isothermal
        self.sie_sersic = EPL(light=source, **self.deflector_dict)

        self.deflector_dict = {
            "vel_disp": 200,
            "gamma_pl": 2.1,
            "e1": 0.1,
            "e2": -0.1,
        }
        # gamma_pl not given, hence using isothermal
        self.epl_sersic = EPL(light=source, **self.deflector_dict)

    def test_redshift(self):
        z = self.sie_sersic._light.redshift
        assert self.kwargs_source["z"] == z

    def test_velocity_dispersion(self):
        vel_disp = self.sie_sersic.velocity_dispersion()
        assert vel_disp == self.deflector_dict["vel_disp"]

    def test_mass_model_lenstronomy_sie(self):
        # Should yeld SIE model as gamma = 2
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.sie_sersic._light.redshift, z_source=2.0
        )
        lens_mass_model_list, kwargs_lens_mass = self.sie_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo,
            spherical=False,
        )
        assert len(lens_mass_model_list) == 1
        assert lens_mass_model_list[0] == "SIE"

        lens_mass_model_list, kwargs_lens_mass = self.sie_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo,
            spherical=True,
        )
        assert len(lens_mass_model_list) == 1
        assert lens_mass_model_list[0] == "SIS"

        lens_mass_model_list, kwargs_lens_mass = self.epl_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo,
            spherical=False,
        )
        assert len(lens_mass_model_list) == 1
        assert lens_mass_model_list[0] == "EPL"

        lens_mass_model_list, kwargs_lens_mass = self.epl_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo,
            spherical=True,
        )
        assert len(lens_mass_model_list) == 1
        assert lens_mass_model_list[0] == "SPP"

    def test_mass_model_no_lensing(self):
        # case when z_source < z_lens
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.sie_sersic._light.redshift, z_source=0.2
        )
        lens_mass_model_list, kwargs_lens_mass = self.sie_sersic.mass_model_lenstronomy(
            lens_cosmo=lens_cosmo
        )
        assert kwargs_lens_mass[0]["theta_E"] == 0.0

    def test_halo_porperties(self):
        gamma = self.sie_sersic.mass_properties["gamma_pl"]
        assert gamma == 2.0


if __name__ == "__main__":
    pytest.main()
