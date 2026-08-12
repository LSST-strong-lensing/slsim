import pytest
import numpy.testing as npt
from slsim.Deflectors.deflector import Deflector
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from slsim.Deflectors.deflector_util import critical_curves_caustics_list


class TestDeflector(object):
    """Testing the Deflector class."""

    def setup_method(self):

        kwargs_light = {
            "extended_source_type": "single_sersic",
            "n_sersic": 1,
            "e1": 0.1,
            "e2": 0.2,
            "angular_size": 0.5,
            "mag_r": 25.0,
            "mag_g": 20.0,
            "stellar_mass": 1e11,
        }
        kwargs_mass = {
            "mass_type": "EPL",
            "vel_disp": 250,
            "gamma_pl": 2,
            "e1": 0.1,
            "e2": -0.1,
        }
        kwargs_mass2 = {
            "mass_type": "EPL",
            "theta_E": 0.8,
            "gamma_pl": 2,
            "e1": 0.1,
            "e2": -0.1,
        }
        self.kwargs_light = kwargs_light
        self.kwargs_mass = kwargs_mass

        self.deflector = Deflector(
            z=0.5,
            center_x=0.1,
            center_y=0,
            kwargs_light=kwargs_light,
            kwargs_mass=kwargs_mass,
        )
        self.deflector3 = Deflector(
            z=0.5,
            center_x=0.1,
            center_y=0,
            kwargs_light=kwargs_light,
            kwargs_mass=kwargs_mass,
        )

        self.deflector2 = Deflector(
            z=0.5,
            center_x=0.1,
            center_y=0,
            kwargs_light=kwargs_light,
            kwargs_mass=kwargs_mass2,
        )
        self.lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5)

        kwargs_mass_nfw = {
            "mass_type": "NFW_HERNQUIST",
            "halo_mass": 10**13,
            "concentration": 10,
            "e1": 0.1,
            "e2": -0.1,
        }
        kwargs_mass_her = {"mass_type": "HERNQUIST"}
        kwargs_pj = {
            "mass_type": "PJAFFE",
            "r_s": 0.1,
            "r_a": 2,
            "vel_disp": 200,
            "e1": 0,
            "e2": 0,
        }

        kwargs_light_hernquist = {
            "extended_source_type": "hernquist",
            "e1": -0.1,
            "e2": 0.1,
            "stellar_mass": 1e11,
            "angular_size": 3.0,
            "mag_g": 20,
        }

        self.deflector_nfw_her = Deflector(
            z=0.5,
            center_x=0,
            center_y=0,
            kwargs_mass=kwargs_mass_nfw,
            kwargs_light=kwargs_light_hernquist,
        )
        self.deflector_backup = Deflector(
            z=0.5, kwargs_mass=kwargs_mass_nfw, kwargs_light=kwargs_light_hernquist
        )

        self.deflector_epl = Deflector(
            z=0.5, center_x=0.1, center_y=0, kwargs_light=None, kwargs_mass=kwargs_mass
        )
        self.deflector_her = Deflector(
            z=0.5, kwargs_mass=kwargs_mass_her, kwargs_light=kwargs_light_hernquist
        )
        self.deflector_jaffe = Deflector(
            z=0.5, kwargs_mass=kwargs_pj, kwargs_light=kwargs_light
        )

    def test_light_ellipticity(self):
        e1_light, e2_light = self.deflector.light_ellipticity
        print(e1_light, e2_light)
        assert pytest.approx(e1_light, rel=1e-3) == self.kwargs_light["e1"]
        assert pytest.approx(e2_light, rel=1e-3) == self.kwargs_light["e2"]

    def test_mass_ellipticity(self):
        e1_mass, e2_mass = self.deflector.mass_ellipticity
        assert pytest.approx(e1_mass, rel=1e-3) == self.kwargs_mass["e1"]
        assert pytest.approx(e2_mass, rel=1e-3) == self.kwargs_mass["e2"]

    def test_init(self):
        with npt.assert_raises(ValueError):
            Deflector(z=0.5, kwargs_mass={"mass_type": "WRONG"})

    def test_magnitude(self):
        band = "g"
        deflector_magnitude = self.deflector.magnitude(band)
        assert isinstance(deflector_magnitude, float)
        assert pytest.approx(deflector_magnitude, rel=1e-3) == 20

    def test_redshift(self):
        z = self.deflector.redshift
        assert pytest.approx(z, rel=1e-3) == 0.5

    def test_velocity_dispersion(self):
        sigma_v = self.deflector.velocity_dispersion(cosmo=None)
        sigma_v2 = self.deflector2.velocity_dispersion(cosmo=None)
        assert pytest.approx(sigma_v, rel=1e-3) == 250
        assert sigma_v2 is None

    def test_deflector_center(self):
        center = self.deflector.deflector_center
        assert isinstance(center[0], float)
        assert isinstance(center[1], float)

    def test_stellar_mass(self):
        stellar_mass = self.deflector.stellar_mass
        npt.assert_almost_equal(stellar_mass, 1e11, decimal=-3)

    def test_light_model_lenstronomy(self):
        band = "g"
        light_model, kwargs_lens_light = self.deflector.light_model_lenstronomy(
            band=band
        )
        assert light_model[0] == "SERSIC_ELLIPSE"
        assert kwargs_lens_light[0]["R_sersic"] == 0.5

    def test_mass_model_lenstronomy(self):
        results = self.deflector.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)[1]
        results2 = self.deflector2.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)[1]
        npt.assert_almost_equal(results[0]["theta_E"], 1.0188773987406035, decimal=7)
        assert results2[0]["theta_E"] == 0.8

    def test_surface_brightness(self):
        # TODO:
        ra, dec = 0, 0
        band = "g"
        r_eff = 1
        mag_arcsec2_center = self.deflector.surface_brightness(ra, dec, band=band)
        mag_arcsec2_r_eff = self.deflector.surface_brightness(
            ra + r_eff, dec, band=band
        )
        # TODO: define a more meaningful test
        npt.assert_almost_equal(
            mag_arcsec2_center / mag_arcsec2_r_eff, 0.8558993, decimal=3
        )

    def test_angular_size(self):
        ang_size = self.deflector.angular_size_light
        npt.assert_almost_equal(ang_size, self.kwargs_light["angular_size"])

    def test_theta_e_when_source_infinity(self):
        try:
            import jax

            print(jax.__path__)

            use_jax = True
        except ImportError:
            use_jax = False
        theta_E_infinity = self.deflector.theta_e_infinity(cosmo=None, use_jax=use_jax)
        assert theta_E_infinity < 15
        theta_E_infinity_new = self.deflector.theta_e_infinity(
            cosmo=None, use_jax=False
        )
        npt.assert_almost_equal(theta_E_infinity, theta_E_infinity_new, decimal=5)
        npt.assert_almost_equal(theta_E_infinity, 1.8024, decimal=3)

        theta_E_infinity_her = self.deflector_nfw_her.theta_e_infinity(
            cosmo=None, use_jax=use_jax
        )
        npt.assert_almost_equal(theta_E_infinity_her, 1.228, decimal=3)

        # we do call the definition twice with use_jax=False to make sure it increases test coverage
        self.deflector_backup.theta_e_infinity(cosmo=None, use_jax=False)
        npt.assert_almost_equal(theta_E_infinity, 1.8024, decimal=2)
        self.deflector_backup.theta_e_infinity(cosmo=None, use_jax=False)
        npt.assert_almost_equal(theta_E_infinity, 1.8024, decimal=2)

        npt.assert_almost_equal(theta_E_infinity, theta_E_infinity_new, decimal=5)

    def test_critical_curves_caustics(self):
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = (
            critical_curves_caustics_list(
                self.deflector_nfw_her,
                10,
                None,
                {
                    "compute_window": 40,
                    "grid_scale": 1.0,
                },
            )
        )

        assert len(ra_crit_list) > 0
        assert len(dec_crit_list) > 0
        assert len(ra_caustic_list) > 0
        assert len(dec_caustic_list) > 0
