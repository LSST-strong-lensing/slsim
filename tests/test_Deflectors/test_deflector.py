import pytest
import numpy.testing as npt
import os
from slsim.Deflectors.deflector import Deflector
from astropy.table import Table
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class TestDeflector(object):
    """Testing the Deflector class."""

    def setup_method(self):
        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        red_one = Table.read(
            os.path.join(module_path, "TestData/red_one_modified.fits"), format="fits"
        )
        self.deflector = Deflector(deflector_type="EPL_SERSIC", **red_one)

        red_two = Table(red_one).copy()
        red_two.remove_column("vel_disp")
        red_two["theta_E"] = 0.8
        self.deflector2 = Deflector(deflector_type="EPL_SERSIC", **red_two)
        self.lens_cosmo = LensCosmo(z_lens=red_two["z"], z_source=1.5)

        deflector_nfw_dict = {
            "halo_mass": 10**13,
            "halo_mass_acc": 0.0,
            "concentration": 10,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "stellar_mass": 10e11,
            "angular_size": 0.001 / 4.84813681109536e-06,
            "e1_light": -0.1,
            "e2_light": 0.1,
            "z": 0.5,
            "mag_g": -20,
        }
        self.deflector_nfw = Deflector(
            deflector_type="NFW_HERNQUIST", **deflector_nfw_dict
        )
        self.deflector_backup = Deflector(
            deflector_type="NFW_HERNQUIST", **deflector_nfw_dict
        )

        self.deflector_epl = Deflector(deflector_type="EPL", **red_two)

        deflector_nfw_cluster_dict = {
            "halo_mass": 10**14,
            "concentration": 10,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "z": 0.3,
            "subhalos": [red_one, red_one],
        }

        self.deflector3 = Deflector(deflector_type="EPL_SERSIC", **red_one)

        self.deflector_nfw2 = Deflector(
            deflector_type="NFW_HERNQUIST", **deflector_nfw_dict
        )
        self.deflector_nfw_cluster1 = Deflector(
            deflector_type="NFW_CLUSTER", **deflector_nfw_cluster_dict
        )
        self.deflector_nfw_cluster2 = Deflector(
            deflector_type="NFW_CLUSTER", **deflector_nfw_cluster_dict
        )

    def test_light_ellipticity(self):
        e1_light, e2_light = self.deflector.light_ellipticity
        assert pytest.approx(e1_light, rel=1e-3) == -0.05661955320450283
        assert pytest.approx(e2_light, rel=1e-3) == 0.08738390223219591

    def test_mass_ellipticity(self):
        e1_mass, e2_mass = self.deflector.mass_ellipticity
        assert pytest.approx(e1_mass, rel=1e-3) == -0.08434700688970058
        assert pytest.approx(e2_mass, rel=1e-3) == 0.09710653297997263

    def test_init(self):
        with npt.assert_raises(ValueError):
            Deflector(deflector_type="WRONG_MODEL", deflector_dict={})

    def test_magnitude(self):
        band = "g"
        deflector_magnitude = self.deflector.magnitude(band)
        assert isinstance(deflector_magnitude[0], float)
        assert pytest.approx(deflector_magnitude[0], rel=1e-3) == 26.4515655

    def test_redshift(self):
        z = self.deflector.redshift
        assert pytest.approx(z, rel=1e-3) == 0.9194649297646337

    def test_velocity_dispersion(self):
        sigma_v = self.deflector.velocity_dispersion(cosmo=None)
        sigma_v2 = self.deflector2.velocity_dispersion(cosmo=None)
        assert pytest.approx(sigma_v, rel=1e-3) == 191.40371531030243
        assert sigma_v2 is None

    def test_deflector_center(self):
        center = self.deflector.deflector_center
        assert isinstance(center[0], float)
        assert isinstance(center[1], float)

    def test_stellar_mass(self):
        stellar_mass = self.deflector.stellar_mass
        npt.assert_almost_equal(stellar_mass, 6.94160421e10, decimal=-3)

    def test_light_model_lenstronomy(self):
        band = "g"
        light_model, kwargs_lens_light = self.deflector.light_model_lenstronomy(
            band=band
        )
        assert light_model[0] == "SERSIC_ELLIPSE"
        assert kwargs_lens_light[0]["R_sersic"] == 7.613175197518637e-07

    def test_mass_model_lenstronomy(self):
        results = self.deflector.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)[1]
        results2 = self.deflector2.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)[1]
        npt.assert_almost_equal(results[0]["theta_E"][0], 0.30360748, decimal=7)
        assert results2[0]["theta_E"] == 0.8

    def test_surface_brightness(self):
        # TODO:
        ra, dec = 0, 0
        band = "g"
        r_eff = 1
        deflector_dict = {
            "vel_disp": 200,
            "e1_mass": 0,
            "e2_mass": 0,
            "stellar_mass": 10**10,
            "z": 0.5,
            "e1_light": 0,
            "e2_light": 0,
            "center_x": 0,
            "center_y": 0,
            "mag_g": 17,
            "angular_size": r_eff,
            "n_sersic": 1,
        }

        deflector = Deflector(deflector_type="EPL_SERSIC", **deflector_dict)
        mag_arcsec2_center = deflector.surface_brightness(ra, dec, band=band)
        mag_arcsec2_r_eff = deflector.surface_brightness(ra + r_eff, dec, band=band)
        # TODO: define a more meaningful test
        npt.assert_almost_equal(
            mag_arcsec2_center / mag_arcsec2_r_eff, 0.9079, decimal=3
        )

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

        theta_E_infinity = self.deflector_nfw.theta_e_infinity(
            cosmo=None, use_jax=use_jax
        )
        # we do call the definition twice with use_jax=False to make sure it increases test coverage
        self.deflector_backup.theta_e_infinity(cosmo=None, use_jax=False)
        npt.assert_almost_equal(theta_E_infinity, 1, decimal=2)
        self.deflector_backup.theta_e_infinity(cosmo=None, use_jax=False)
        npt.assert_almost_equal(theta_E_infinity, 1, decimal=2)

        # Test the multi_plane case
        # EPL_SERSIC
        theta_E_infinity = self.deflector3.theta_e_infinity(
            cosmo=None, multi_plane=True
        )
        assert theta_E_infinity < 15
        theta_E_infinity_new = self.deflector3.theta_e_infinity(
            cosmo=None, multi_plane=True
        )
        npt.assert_almost_equal(theta_E_infinity, theta_E_infinity_new, decimal=5)

        # NFW_CLUSTER
        theta_E_infinity = self.deflector_nfw_cluster1.theta_e_infinity(
            cosmo=None, use_jax=use_jax
        )
        assert theta_E_infinity < 30
        theta_E_infinity_multi = self.deflector_nfw_cluster2.theta_e_infinity(
            cosmo=None, multi_plane=True, use_jax=use_jax
        )
        assert theta_E_infinity_multi < 30

        # NFW_HERNQUIST
        theta_E_infinity = self.deflector_nfw2.theta_e_infinity(
            cosmo=None, multi_plane=True, use_jax=use_jax
        )
        assert theta_E_infinity < 30
