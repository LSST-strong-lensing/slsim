import pytest
import numpy as np
from numpy import testing as npt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from slsim.lens import (
    Lens,
    image_separation_from_positions,
    theta_e_when_source_infinity,
)
import os


class TestLens(object):
    # pytest.fixture(scope='class')
    def setup_method(self):
        # path = os.path.dirname(slsim.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "TestData/blue_one_modified.fits"), format="fits"
        )
        red_one = Table.read(
            os.path.join(path, "TestData/red_one_modified.fits"), format="fits"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        while True:
            gg_lens = Lens(
                source_dict=self.source_dict,
                deflector_dict=self.deflector_dict,
                lens_equation_solver="lenstronomy_analytical",
                cosmo=cosmo,
            )
            if gg_lens.validity_test():
                self.gg_lens = gg_lens
                break

    def test_nfw_hernquist_lens(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "TestData/blue_one_modified.fits"), format="fits"
        )
        source_dict = blue_one
        deflector_dict = {
            "halo_mass": 10**13,
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

        while True:
            gg_lens = Lens(
                source_dict=source_dict,
                deflector_dict=deflector_dict,
                deflector_type="NFW_HERNQUIST",
                lens_equation_solver="lenstronomy_default",
                cosmo=cosmo,
            )
            if gg_lens.validity_test():
                # self.gg_lens = gg_lens
                break

    def test_deflector_ellipticity(self):
        e1_light, e2_light, e1_mass, e2_mass = self.gg_lens.deflector_ellipticity()
        assert pytest.approx(e1_light, rel=1e-3) == -0.05661955320450283
        assert pytest.approx(e2_light, rel=1e-3) == 0.08738390223219591
        assert pytest.approx(e1_mass, rel=1e-3) == -0.08434700688970058
        assert pytest.approx(e2_mass, rel=1e-3) == 0.09710653297997263

    def test_deflector_magnitude(self):
        band = "g"
        deflector_magnitude = self.gg_lens.deflector_magnitude(band)
        assert isinstance(deflector_magnitude[0], float)
        assert pytest.approx(deflector_magnitude[0], rel=1e-3) == 26.4515655

    def test_source_magnitude(self):
        band = "g"
        source_magnitude = self.gg_lens.extended_source_magnitude(band)
        source_magnitude_lensed = self.gg_lens.extended_source_magnitude(
            band, lensed=True
        )
        host_mag = self.gg_lens.extended_source_magnification()
        expected_lensed_mag = source_magnitude - 2.5 * np.log10(host_mag)
        assert pytest.approx(source_magnitude[0], rel=1e-3) == 30.780194
        assert source_magnitude_lensed == expected_lensed_mag

    def test_image_separation_from_positions(self):
        image_positions = self.gg_lens.extended_source_image_positions()
        image_separation = image_separation_from_positions(image_positions)
        theta_E_infinity = theta_e_when_source_infinity(
            deflector_dict=self.deflector_dict
        )
        assert image_separation < 2 * theta_E_infinity

    def test_theta_e_when_source_infinity(self):
        theta_E_infinity = theta_e_when_source_infinity(
            deflector_dict=self.deflector_dict
        )
        # We expect that theta_E_infinity should be less than 15
        assert theta_E_infinity < 15

    def test_extended_source_magnification(self):
        host_mag = self.gg_lens.extended_source_magnification()
        assert host_mag > 0

    def test_deflector_stellar_mass(self):
        s_mass = self.gg_lens.deflector_stellar_mass()
        assert s_mass >= 10**5

    def test_deflector_velocity_dispersion(self):
        vdp = self.gg_lens.deflector_velocity_dispersion()
        assert vdp >= 10

    def test_los_linear_distortions(self):
        losd = self.gg_lens.los_linear_distortions()
        assert losd != 0

    def test_point_source_arrival_times(self):
        dt_days = self.gg_lens.point_source_arrival_times()
        assert np.min(dt_days) > -1000
        assert np.max(dt_days) < 1000

    def test_image_observer_times(self):
        t_obs = 1000
        t_obs2 = np.array([100, 200, 300])
        dt_days = self.gg_lens.image_observer_times(t_obs=t_obs)
        dt_days2 = self.gg_lens.image_observer_times(t_obs=t_obs2)
        arrival_times = self.gg_lens.point_source_arrival_times()
        observer_times = (t_obs + arrival_times - np.min(arrival_times))[:, np.newaxis]
        observer_times2 = (
            t_obs2[:, np.newaxis] + arrival_times - np.min(arrival_times)
        ).T
        npt.assert_almost_equal(dt_days, observer_times, decimal=5)
        npt.assert_almost_equal(dt_days2, observer_times2, decimal=5)

    def test_deflector_light_model_lenstronomy(self):
        kwargs_lens_light = self.gg_lens.deflector_light_model_lenstronomy(band="g")
        assert len(kwargs_lens_light) >= 1

    def test_lens_equation_solver(self):
        """Tests analytical and numerical lens equation solver options."""
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        gg_lens = Lens(
            lens_equation_solver="lenstronomy_default",
            source_dict=self.source_dict,
            deflector_dict=self.deflector_dict,
            cosmo=cosmo,
        )
        while True:
            gg_lens.validity_test()
            break

        gg_lens = Lens(
            lens_equation_solver="lenstronomy_analytical",
            source_dict=self.source_dict,
            deflector_dict=self.deflector_dict,
            cosmo=cosmo,
        )
        while True:
            gg_lens.validity_test()
            break


@pytest.fixture
def pes_lens_instance():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "TestData/source_dict_ps.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "TestData/deflector_dict_ps.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        pes_lens = Lens(
            source_dict=source_dict,
            deflector_dict=deflector_dict,
            source_type="point_plus_extended",
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
            cosmo=cosmo,
        )
        if pes_lens.validity_test():
            pes_lens = pes_lens
            break
    return pes_lens


def test_point_source_magnitude(pes_lens_instance):
    pes_lens = pes_lens_instance
    mag = pes_lens.point_source_magnitude(band="i", lensed=True)
    mag_unlensed = pes_lens.point_source_magnitude(band="i")
    assert len(mag) >= 2
    assert len(mag_unlensed) == 1


@pytest.fixture
def supernovae_lens_instance():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "TestData/supernovae_source_dict.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "TestData/supernovae_deflector_dict.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        supernovae_lens = Lens(
            source_dict=source_dict,
            deflector_dict=deflector_dict,
            source_type="point_plus_extended",
            variability_model="light_curve",
            kwargs_variability={"MJD", "ps_mag_r"},
            cosmo=cosmo,
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens


def test_point_source_magnitude_with_lightcurve(supernovae_lens_instance):
    supernovae_lens = supernovae_lens_instance
    mag = supernovae_lens.point_source_magnitude(band="r", lensed=True)
    expected_results = supernovae_lens_instance.source.source_dict["ps_mag_r"]
    assert mag[0][0] != expected_results[0][0]
    assert mag[1][0] != expected_results[0][0]


if __name__ == "__main__":
    pytest.main()
