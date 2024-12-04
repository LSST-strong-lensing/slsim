import copy

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
from slsim.LOS.los_individual import LOSIndividual
from slsim.LOS.los_pop import LOSPop
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
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
        blue_one["angular_size"] = blue_one["angular_size"] / 4.84813681109536e-06
        red_one = Table.read(
            os.path.join(path, "TestData/red_one_modified.fits"), format="fits"
        )
        red_one["angular_size"] = red_one["angular_size"] / 4.84813681109536e-06
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        self.los_individual = LOSIndividual(kappa=0.1, gamma=[-0.1, -0.2])

        mag_arc_limit = {"i": 35, "g": 35, "r": 35}
        while True:
            self.source = Source(
                source_dict=self.source_dict,
                cosmo=cosmo,
                source_type="extended",
                light_profile="single_sersic",
            )
            self.deflector = Deflector(
                deflector_type="EPL",
                deflector_dict=self.deflector_dict,
            )
            gg_lens = Lens(
                source_class=self.source,
                deflector_class=self.deflector,
                los_class=self.los_individual,
                lens_equation_solver="lenstronomy_analytical",
                # kwargs_variability={"MJD", "ps_mag_i"},  # This line will not be used in
                # the testing but at least code go through this warning message.
                cosmo=cosmo,
            )
            if gg_lens.validity_test(mag_arc_limit=mag_arc_limit):
                self.gg_lens = gg_lens
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
        image_positions = self.gg_lens.extended_source_image_positions()[0]
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
        host_mag = self.gg_lens.extended_source_magnification()[0]
        assert host_mag > 0

    def test_deflector_stellar_mass(self):
        s_mass = self.gg_lens.deflector_stellar_mass()
        assert s_mass >= 10**5

    def test_deflector_velocity_dispersion(self):
        vdp = self.gg_lens.deflector_velocity_dispersion()
        assert vdp >= 10

    def test_los_linear_distortions(self):
        kappa, gamma1, gamma2 = self.gg_lens.los_linear_distortions
        assert kappa == self.los_individual.convergence
        g1, g2 = self.los_individual.shear
        assert gamma1 == g1
        assert gamma2 == g2

    def test_point_source_arrival_times(self):
        dt_days = self.gg_lens.point_source_arrival_times()
        assert np.min(dt_days) > -1000
        assert np.max(dt_days) < 1000

    def test_image_observer_times(self):
        t_obs = 1000
        t_obs2 = np.array([100, 200, 300])
        dt_days = self.gg_lens.image_observer_times(t_obs=t_obs)
        dt_days2 = self.gg_lens.image_observer_times(t_obs=t_obs2)
        arrival_times = self.gg_lens.point_source_arrival_times()[0]
        observer_times = (t_obs - arrival_times + np.min(arrival_times))[:, np.newaxis]
        observer_times2 = (
            t_obs2[:, np.newaxis] - arrival_times + np.min(arrival_times)
        ).T
        npt.assert_almost_equal(dt_days, observer_times, decimal=5)
        npt.assert_almost_equal(dt_days2, observer_times2, decimal=5)

    def test_deflector_light_model_lenstronomy(self):
        kwargs_lens_light = self.gg_lens.deflector_light_model_lenstronomy(band="g")
        assert len(kwargs_lens_light) >= 1

    def test_lens_equation_solver(self):
        # Tests analytical and numerical lens equation solver options.
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        gg_lens = Lens(
            lens_equation_solver="lenstronomy_default",
            source_class=self.source,
            deflector_class=self.deflector,
            cosmo=cosmo,
        )
        while True:
            gg_lens.validity_test()
            break

        gg_lens = Lens(
            lens_equation_solver="lenstronomy_analytical",
            source_class=self.source,
            deflector_class=self.deflector,
            cosmo=cosmo,
        )
        while True:
            gg_lens.validity_test()
            break

        # and here for NFW-Hernquist model
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "TestData/blue_one_modified.fits"), format="fits"
        )
        source_dict = blue_one
        deflector_dict = {
            "halo_mass": 10**13.8,
            "concentration": 10,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "stellar_mass": 10.5e11,
            "angular_size": 0.16,
            "e1_light": -0.1,
            "e2_light": 0.1,
            "z": 0.5,
            "mag_g": -20,
        }

        while True:
            self.source2 = Source(
                source_dict=source_dict,
                cosmo=cosmo,
                source_type="extended",
                light_profile="single_sersic",
            )
            self.deflector2 = Deflector(
                deflector_type="NFW_HERNQUIST",
                deflector_dict=deflector_dict,
            )
            gg_lens = Lens(
                source_class=self.source2,
                deflector_class=self.deflector2,
                lens_equation_solver="lenstronomy_default",
                cosmo=cosmo,
            )
            if gg_lens.validity_test():
                # self.gg_lens = gg_lens
                break

        # here for NFW-Cluster model
        subhalos_table = Table.read(
            os.path.join(path, "TestData/subhalos_table.fits"), format="fits"
        )
        source_dict = blue_one
        deflector_dict = {
            "halo_mass": 10**14,
            "concentration": 5,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "z": 0.42,
            "subhalos": subhalos_table,
        }
        while True:
            self.source3 = Source(
                source_dict=source_dict,
                cosmo=cosmo,
                source_type="extended",
                light_profile="single_sersic",
            )
            self.deflector3 = Deflector(
                deflector_type="NFW_CLUSTER",
                deflector_dict=deflector_dict,
            )
            cg_lens = Lens(
                source_class=self.source3,
                deflector_class=self.deflector3,
                lens_equation_solver="lenstronomy_default",
                cosmo=cosmo,
            )
            if cg_lens.validity_test(max_image_separation=50.0):
                break

    def test_kappa_star(self):

        from lenstronomy.Util.util import make_grid

        delta_pix = 0.05
        x, y = make_grid(numPix=200, deltapix=delta_pix)
        kappa_star = self.gg_lens.kappa_star(x, y)
        stellar_mass_from_kappa_star = (
            np.sum(kappa_star)
            * delta_pix**2
            * self.gg_lens._lens_cosmo.sigma_crit_angle
        )
        stellar_mass = self.gg_lens.deflector_stellar_mass()
        npt.assert_almost_equal(
            stellar_mass_from_kappa_star / stellar_mass, 1, decimal=1
        )


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
        source4 = Source(
            source_dict=source_dict,
            cosmo=cosmo,
            source_type="point_plus_extended",
            light_profile="single_sersic",
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
        )
        deflector4 = Deflector(
            deflector_type="EPL",
            deflector_dict=deflector_dict,
        )
        pes_lens = Lens(
            source_class=source4,
            deflector_class=deflector4,
            cosmo=cosmo,
        )
        if pes_lens.validity_test():
            pes_lens = pes_lens
            break
    return pes_lens


def test_point_source_magnitude(pes_lens_instance):
    pes_lens = pes_lens_instance
    mag = pes_lens.point_source_magnitude(band="i", lensed=True)[0]
    mag_unlensed = pes_lens.point_source_magnitude(band="i")[0]
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
        source5 = Source(
            source_dict=source_dict,
            cosmo=cosmo,
            source_type="point_plus_extended",
            light_profile="single_sersic",
            variability_model="light_curve",
            kwargs_variability={"MJD", "ps_mag_r"},
        )
        deflector5 = Deflector(
            deflector_type="EPL",
            deflector_dict=deflector_dict,
        )
        supernovae_lens = Lens(
            source_class=source5,
            deflector_class=deflector5,
            cosmo=cosmo,
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens


def test_point_source_magnitude_with_lightcurve(supernovae_lens_instance):
    supernovae_lens = supernovae_lens_instance
    mag = supernovae_lens.point_source_magnitude(band="r", lensed=True)[0]
    expected_results = supernovae_lens_instance.source[0].source_dict["ps_mag_r"]
    assert mag[0][0] != expected_results[0][0]
    assert mag[1][0] != expected_results[0][0]


class TestDifferentLens(object):
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
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        self.source6 = Source(
            source_dict=self.source_dict,
            cosmo=self.cosmo,
            source_type="extended",
            light_profile="single_sersic",
        )
        self.deflector6 = Deflector(
            deflector_type="EPL",
            deflector_dict=self.deflector_dict,
        )

    def test_different_setting(self):
        los1 = LOSPop(
            los_bool=True,
            mixgauss_gamma=True,
            nonlinear_los_bool=False,
        )
        gg_lens = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los1.draw_los(
                source_redshift=self.source6.redshift,
                deflector_redshift=self.deflector6.redshift,
            ),
        )
        assert gg_lens.external_shear >= 0
        assert isinstance(gg_lens.external_convergence, float)
        assert isinstance(gg_lens.external_shear, float)

        los2 = LOSPop(
            los_bool=True,
            mixgauss_gamma=False,
            nonlinear_los_bool=True,
        )

        gg_lens_2 = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los2.draw_los(
                source_redshift=self.source6.redshift,
                deflector_redshift=self.deflector6.redshift,
            ),
        )
        assert gg_lens_2.external_shear >= 0
        assert isinstance(gg_lens_2.external_convergence, float)
        assert isinstance(gg_lens_2.external_shear, float)

        los3 = LOSPop(los_bool=False)
        gg_lens_3 = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los3.draw_los(
                source_redshift=self.source6.redshift,
                deflector_redshift=self.deflector6.redshift,
            ),
        )
        assert gg_lens_3.external_convergence == 0
        assert gg_lens_3.external_shear == 0

        los4 = LOSPop(
            los_bool=True,
            mixgauss_gamma=True,
            nonlinear_los_bool=True,
        )
        with pytest.raises(ValueError):
            gg_lens_4 = Lens(
                source_class=self.source6,
                deflector_class=self.deflector6,
                cosmo=self.cosmo,
                los_class=los4.draw_los(
                    deflector_redshift=self.deflector6.redshift,
                    source_redshift=self.source6.redshift,
                ),
            )
            gg_lens_4.external_convergence()

    def test_image_number(self):
        los = LOSIndividual(kappa=0, gamma=[0, 0])
        gg_lens_number = Lens(
            source_class=self.source6,
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los,
        )
        image_number = gg_lens_number.image_number
        assert (
            (image_number[0] == 4) or (image_number[0] == 2) or (image_number[0] == 1)
        )

        gg_lens_multisource = Lens(
            source_class=[self.source6, self.source6],
            deflector_class=self.deflector6,
            cosmo=self.cosmo,
            los_class=los,
        )
        kwargs_model = gg_lens_multisource.lenstronomy_kwargs()[0]
        kwargs_model_keys = kwargs_model.keys()
        expected_kwargs_model = [
            "lens_light_model_list",
            "lens_model_list",
            "z_lens",
            "lens_redshift_list",
            "source_redshift_list",
            "z_source_convention",
            "cosmo",
            "source_light_model_list",
        ]
        assert expected_kwargs_model[0] in kwargs_model_keys
        assert expected_kwargs_model[1] in kwargs_model_keys
        assert expected_kwargs_model[2] in kwargs_model_keys
        assert expected_kwargs_model[3] in kwargs_model_keys
        assert expected_kwargs_model[4] in kwargs_model_keys
        assert expected_kwargs_model[5] in kwargs_model_keys
        assert expected_kwargs_model[6] in kwargs_model_keys


@pytest.fixture
def supernovae_lens_instance_double_sersic_multisource():
    path = os.path.dirname(__file__)
    source_dict = Table.read(
        os.path.join(path, "TestData/source_supernovae_new.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(path, "TestData/deflector_supernovae_new.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        source = Source(
            source_dict=source_dict,
            cosmo=cosmo,
            source_type="point_plus_extended",
            light_profile="double_sersic",
            lightcurve_time=np.linspace(-20, 100, 1000),
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_type="Ia",
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
        )
        deflector = Deflector(
            deflector_type="EPL",
            deflector_dict=deflector_dict,
        )
        supernovae_lens = Lens(
            deflector_class=deflector,
            source_class=[source, source],
            cosmo=cosmo,
        )
        if supernovae_lens.validity_test():
            supernovae_lens = supernovae_lens
            break
    return supernovae_lens


class TestMultiSource(object):
    def setup_method(self):
        np.random.seed(42)
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        path = os.path.dirname(__file__)
        source_dict1 = Table.read(
            os.path.join(path, "TestData/source_supernovae_new.fits"), format="fits"
        )
        deflector_dict = Table.read(
            os.path.join(path, "TestData/deflector_supernovae_new.fits"), format="fits"
        )

        deflector_dict_ = dict(zip(deflector_dict.colnames, deflector_dict[0]))
        self.gamma_pl = 1.8
        deflector_dict_["gamma_pl"] = self.gamma_pl
        source_dict2 = copy.deepcopy(source_dict1)
        source_dict2["z"] += 2
        self.source1 = Source(
            source_dict=source_dict2,
            cosmo=self.cosmo,
            source_type="point_plus_extended",
            light_profile="double_sersic",
            lightcurve_time=np.linspace(-20, 100, 1000),
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_type="Ia",
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
        )
        # We initiate the another Source class with the same source. In this class,
        # source position will be different and all the lensing quantities will be different
        self.source2 = Source(
            source_dict=source_dict1,
            cosmo=self.cosmo,
            source_type="point_plus_extended",
            light_profile="double_sersic",
            lightcurve_time=np.linspace(-20, 100, 1000),
            variability_model="light_curve",
            kwargs_variability={"supernovae_lightcurve", "i"},
            sn_type="Ia",
            sn_absolute_mag_band="bessellb",
            sn_absolute_zpsys="ab",
        )
        self.deflector = Deflector(
            deflector_type="EPL",
            deflector_dict=deflector_dict_,
            sis_convention=False,
        )

        self.lens_class1 = Lens(
            deflector_class=self.deflector,
            source_class=self.source1,
            cosmo=self.cosmo,
        )
        self.lens_class2 = Lens(
            deflector_class=self.deflector,
            source_class=self.source2,
            cosmo=self.cosmo,
        )
        self.lens_class3 = Lens(
            deflector_class=self.deflector,
            source_class=[self.source1, self.source2],
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_general",
        )

        self.lens_class3_analytical = Lens(
            deflector_class=self.deflector,
            source_class=[self.source1, self.source2],
            cosmo=self.cosmo,
            lens_equation_solver="lenstronomy_analytical",
        )

    def test_point_source_arrival_time_multi(self):
        gamma_pl_out = self.deflector.halo_properties
        assert gamma_pl_out == self.gamma_pl

        point_source_arival_time1 = self.lens_class1.point_source_arrival_times()
        point_source_arival_time2 = self.lens_class2.point_source_arrival_times()
        point_source_arival_time3 = self.lens_class3.point_source_arrival_times()
        # Test multisource point source arrival time.
        assert np.all(point_source_arival_time1[0]) == np.all(
            point_source_arival_time3[0]
        )
        assert np.all(point_source_arival_time2[0]) == np.all(
            point_source_arival_time3[1]
        )

        point_source_arival_time3 = (
            self.lens_class3_analytical.point_source_arrival_times()
        )
        # Test multisource point source arival time.
        assert np.all(point_source_arival_time1[0]) == np.all(
            point_source_arival_time3[0]
        )
        assert np.all(point_source_arival_time2[0]) == np.all(
            point_source_arival_time3[1]
        )

    def test_ps_magnification_multi(self):
        ps_magnification1 = self.lens_class1.point_source_magnification()
        ps_magnification2 = self.lens_class2.point_source_magnification()
        ps_magnification3 = self.lens_class3.point_source_magnification()
        # Test multisource point source magnifications.
        assert np.all(ps_magnification1[0]) == np.all(ps_magnification3[0])
        assert np.all(ps_magnification2[0]) == np.all(ps_magnification3[1])

        ps_magnification3 = self.lens_class3_analytical.point_source_magnification()
        assert np.all(ps_magnification1[0]) == np.all(ps_magnification3[0])
        assert np.all(ps_magnification2[0]) == np.all(ps_magnification3[1])

    def test_es_magnification_multi(self):
        es_magnification1 = self.lens_class1.extended_source_magnification()
        es_magnification2 = self.lens_class2.extended_source_magnification()
        es_magnification3 = self.lens_class3.extended_source_magnification()
        # Test multisource extended source magnifications.
        npt.assert_almost_equal(
            es_magnification1[0] / es_magnification3[0], 1, decimal=1
        )
        npt.assert_almost_equal(
            es_magnification2[0] / es_magnification3[1], 1, decimal=1
        )

        es_magnification3 = self.lens_class3_analytical.extended_source_magnification()
        npt.assert_almost_equal(
            es_magnification1[0] / es_magnification3[0], 1, decimal=1
        )
        npt.assert_almost_equal(
            es_magnification2[0] / es_magnification3[1], 1, decimal=1
        )

    def test_einstein_radius_multi(self):
        einstein_radius1 = self.lens_class1.einstein_radius
        einstein_radius2 = self.lens_class2.einstein_radius
        einstein_radius3 = self.lens_class3.einstein_radius
        # Test multisource einstein radius.
        npt.assert_almost_equal(einstein_radius1[0], einstein_radius3[0], decimal=2)
        npt.assert_almost_equal(einstein_radius2[0], einstein_radius3[1], decimal=2)

        einstein_radius3 = self.lens_class3_analytical.einstein_radius
        npt.assert_almost_equal(einstein_radius1[0], einstein_radius3[0], decimal=5)
        npt.assert_almost_equal(einstein_radius2[0], einstein_radius3[1], decimal=5)

    def test_image_observer_time_multi(self):
        observation_time = 50
        image_observation_time1 = self.lens_class1.image_observer_times(
            observation_time
        )
        image_observation_time2 = self.lens_class2.image_observer_times(
            observation_time
        )
        image_observation_time3 = self.lens_class3.image_observer_times(
            observation_time
        )
        # Test multisource image observation time
        npt.assert_almost_equal(
            image_observation_time1[0], image_observation_time3[0][0], decimal=5
        )
        # assert image_observation_time1[0] == image_observation_time3[0][0]
        npt.assert_almost_equal(
            image_observation_time2, image_observation_time3[1], decimal=5
        )
        # assert np.all(image_observation_time2 == image_observation_time3[1])
        assert len(self.lens_class3.image_observer_times(t_obs=10)) == 2

        image_observation_time3 = self.lens_class3_analytical.image_observer_times(
            observation_time
        )
        # Test multisource image observation time
        npt.assert_almost_equal(
            image_observation_time1[0], image_observation_time3[0][0], decimal=5
        )
        # assert image_observation_time1[0] == image_observation_time3[0][0]
        npt.assert_almost_equal(
            image_observation_time2, image_observation_time3[1], decimal=5
        )


if __name__ == "__main__":
    pytest.main()
