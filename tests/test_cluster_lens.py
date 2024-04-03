import pytest
import numpy as np
from numpy import testing as npt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from slsim.lens import (
    image_separation_from_positions,
    theta_e_when_source_infinity,
)
from slsim.cluster_lens import ClusterLens
import os


class TestClusterLens(object):
    # pytest.fixture(scope='class')
    def setup_method(self):
        # path = os.path.dirname(slsim.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "TestData/blue_one_modified.fits"), format="fits"
        )
        halo = Table.read(os.path.join(path, "TestData/halo_EPL.fits"), format="fits")
        subhaloes_table = Table.read(
            os.path.join(path, "TestData/subhaloes_table.fits"), format="fits"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = halo
        self.subhaloes_table = subhaloes_table
        while True:
            cg_lens = ClusterLens(
                source_dict=self.source_dict,
                deflector_dict=self.deflector_dict,
                subhaloes_table=self.subhaloes_table,
                lens_equation_solver="lenstronomy_analytical",
                cosmo=cosmo,
            )
            if cg_lens.validity_test():
                self.cg_lens = cg_lens
                break

    def test_deflector_ellipticity(self):
        e1_light, e2_light, e1_mass, e2_mass = self.cg_lens.deflector_ellipticity()
        assert pytest.approx(e1_light, rel=1e-3) == 0.0
        assert pytest.approx(e2_light, rel=1e-3) == 0.0
        assert pytest.approx(e1_mass, rel=1e-3) == -0.08434700688970058
        assert pytest.approx(e2_mass, rel=1e-3) == 0.09710653297997263

    def test_deflector_magnitude(self):
        # TODO: test magnitude of each subhalo
        pass

    def test_source_magnitude(self):
        band = "g"
        source_magnitude = self.cg_lens.extended_source_magnitude(band)
        source_magnitude_lensed = self.cg_lens.extended_source_magnitude(
            band, lensed=True
        )
        host_mag = self.cg_lens.extended_source_magnification()
        expected_lensed_mag = source_magnitude - 2.5 * np.log10(host_mag)
        assert pytest.approx(source_magnitude[0], rel=1e-3) == 30.780194
        assert source_magnitude_lensed == expected_lensed_mag

    def test_image_separation_from_positions(self):
        image_positions = self.cg_lens.extended_source_image_positions()
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
        host_mag = self.cg_lens.extended_source_magnification()
        assert host_mag > 0

    def test_deflector_stellar_mass(self):
        # TODO: test stellar mass of each subhalo
        pass

    def test_deflector_velocity_dispersion(self):
        vdp = self.cg_lens.deflector_velocity_dispersion()
        assert vdp >= 10

    def test_los_linear_distortions(self):
        losd = self.cg_lens.los_linear_distortions()
        assert losd != 0

    def test_point_source_arrival_times(self):
        dt_days = self.cg_lens.point_source_arrival_times()
        # TODO: review limits for cluster scale halo
        assert np.min(dt_days) > -10000
        assert np.max(dt_days) < 10000

    def test_image_observer_times(self):
        t_obs = 1000
        t_obs2 = np.array([100, 200, 300])
        dt_days = self.cg_lens.image_observer_times(t_obs=t_obs)
        dt_days2 = self.cg_lens.image_observer_times(t_obs=t_obs2)
        arrival_times = self.cg_lens.point_source_arrival_times()
        observer_times = (t_obs + arrival_times - np.min(arrival_times))[:, np.newaxis]
        observer_times2 = (
            t_obs2[:, np.newaxis] + arrival_times - np.min(arrival_times)
        ).T
        npt.assert_almost_equal(dt_days, observer_times, decimal=5)
        npt.assert_almost_equal(dt_days2, observer_times2, decimal=5)

    def test_deflector_light_model_lenstronomy(self):
        kwargs_lens_light = self.cg_lens.deflector_light_model_lenstronomy(band="g")
        assert len(kwargs_lens_light) >= 1

    def test_lens_equation_solver(self):
        """Tests analytical and numerical lens equation solver options."""
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        cg_lens = ClusterLens(
            lens_equation_solver="lenstronomy_default",
            source_dict=self.source_dict,
            deflector_dict=self.deflector_dict,
            subhaloes_table=self.subhaloes_table,
            cosmo=cosmo,
        )
        while True:
            cg_lens.validity_test()
            break

        cg_lens = ClusterLens(
            lens_equation_solver="lenstronomy_analytical",
            source_dict=self.source_dict,
            deflector_dict=self.deflector_dict,
            subhaloes_table=self.subhaloes_table,
            cosmo=cosmo,
        )
        while True:
            cg_lens.validity_test()
            break


@pytest.fixture
def pes_lens_instance():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "TestData/source_dict_ps.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "TestData/halo_EPL.fits"), format="fits"
    )
    subhaloes_table = Table.read(
        os.path.join(path, "TestData/subhaloes_table.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        pes_lens = ClusterLens(
            source_dict=source_dict,
            deflector_dict=deflector_dict,
            subhaloes_table=subhaloes_table,
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
        os.path.join(path, "TestData/halo_EPL.fits"), format="fits"
    )
    subhaloes_table = Table.read(
        os.path.join(path, "TestData/subhaloes_table.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        supernovae_lens = ClusterLens(
            source_dict=source_dict,
            deflector_dict=deflector_dict,
            subhaloes_table=subhaloes_table,
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
