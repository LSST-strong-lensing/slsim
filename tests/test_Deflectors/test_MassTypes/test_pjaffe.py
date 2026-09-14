from slsim.Deflectors.MassTypes.pjaffe import PJAFFE
from slsim.Sources.source import Source
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import numpy.testing as npt


class TestPJaffe(object):

    def setup_method(self):

        self.kwargs_light = {
            "extended_source_type": "single_sersic",
            "z": 0.5,
            "mag_r": 20,
            "angular_size": 1,
            "n_sersic": 1,
            "e1": 0,
            "e2": 0,
        }
        light = Source(**self.kwargs_light)

        self.kwargs_mass = {
            "r_s": 0.2,
            "r_a": 2,
            "vel_disp": 250,
            "e1": 0.1,
            "e2": -0.1,
        }

        self.mass = PJAFFE(light=light, **self.kwargs_mass)

    def test_velocity_dispersion(self):
        vel_disp = self.mass.velocity_dispersion(cosmo=None)
        npt.assert_almost_equal(vel_disp, self.kwargs_mass["vel_disp"])

    def test_mass_properties(self):

        kwargs_properties = self.mass.mass_properties
        npt.assert_almost_equal(kwargs_properties["r_s"], self.kwargs_mass["r_s"])

    def test_ellipticity(self):

        e1, e2 = self.mass.ellipticity
        npt.assert_almost_equal(e1, self.kwargs_mass["e1"])
        npt.assert_almost_equal(e2, self.kwargs_mass["e2"])

    def test_mass_model_lenstronomy(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        z_source = 2
        lens_cosmo = LensCosmo(
            z_lens=self.kwargs_light["z"], z_source=z_source, cosmo=cosmo
        )

        lens_mass_model_list, kwargs_lens = self.mass.mass_model_lenstronomy(
            lens_cosmo, spherical=False
        )
        assert lens_mass_model_list[0] == "PJAFFE_ELLIPSE_POTENTIAL"
        lens_mass_model_list, kwargs_lens = self.mass.mass_model_lenstronomy(
            lens_cosmo, spherical=True
        )
        assert lens_mass_model_list[0] == "PJAFFE"
