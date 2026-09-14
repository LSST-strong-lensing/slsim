from slsim.Deflectors.MassTypes.nfw_hernquist import NFWHernquist
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from slsim.Sources.source import Source


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
        light = Source(
            z=0.5,
            extended_source_type="hernquist",
            angular_size=0.001 / 4.84813681109536e-06,
            stellar_mass=1e11,
            e1=-0.1,
            e2=0.1,
            mag_g=20,
        )
        self.deflector_dict = {
            "halo_mass": 10**13,
            "concentration": 10,
            "e1": 0.1,
            "e2": -0.1,
            "light": light,
        }
        self.nfw_hernquist = NFWHernquist(**self.deflector_dict)

    def test_redshift(self):
        z = self.nfw_hernquist._light.redshift
        assert self.deflector_dict["light"].redshift == z

    def test_halo_properties(self):
        kwargs_halo = self.nfw_hernquist.mass_properties
        assert kwargs_halo["halo_mass"] == self.deflector_dict["halo_mass"]
        assert kwargs_halo["concentration"] == self.deflector_dict["concentration"]

    def test_velocity_dispersion(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        vel_disp = self.nfw_hernquist.velocity_dispersion(cosmo=cosmo)
        npt.assert_almost_equal(vel_disp, 176, decimal=-1)
        assert self.nfw_hernquist.velocity_dispersion(cosmo=cosmo) == vel_disp

    def test_light_model_lenstronomy(self):
        lens_light_model_list, kwargs_lens_light = (
            self.nfw_hernquist._light.kwargs_extended_light(band="g")
        )
        assert len(lens_light_model_list) == 1

    def test_mass_model_lenstronomy(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(
            cosmo=cosmo, z_lens=self.deflector_dict["light"].redshift, z_source=2.0
        )
        lens_mass_model_list, kwargs_lens_mass = (
            self.nfw_hernquist.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo, spherical=False
            )
        )
        assert lens_mass_model_list[0] == "NFW_ELLIPSE_CSE"
        assert len(lens_mass_model_list) == 2

        lens_mass_model_list, kwargs_lens_mass = (
            self.nfw_hernquist.mass_model_lenstronomy(
                lens_cosmo=lens_cosmo, spherical=True
            )
        )
        assert lens_mass_model_list[0] == "NFW"
        assert len(lens_mass_model_list) == 2
