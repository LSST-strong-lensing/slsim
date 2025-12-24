from slsim.Halos.halos_plus_glass import (
    read_glass_data,
    generate_samples_from_glass,
    skyarea_form_n,
    generate_maps_kmean_zero_using_halos,
    halos_plus_glass,
    generate_meanzero_halos_multiple_times,
    run_halos_without_kde,
    run_halos_without_kde_by_multiprocessing,
    run_kappaext_gammaext_kde_by_multiprocessing,
    convergence_mean_0,
    run_certain_redshift_lensext_kde_by_multiprocessing,
    run_certain_redshift_many_by_multiprocessing,
    run_total_kappa_by_multiprocessing,
    run_total_mass_by_multiprocessing,
    worker_run_total_kappa_by_multiprocessing,
    worker_run_total_mass_by_multiprocessing,
    worker_certain_redshift_many,
    worker_certain_redshift_lensext_kde,
    worker_kappaext_gammaext_kde,
    worker_run_halos_without_kde,
    run_average_mass_by_multiprocessing,
    worker_run_average_mass_by_multiprocessing,
)
import os
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


class Testhalosplusglass(object):
    def setup_method(self):
        current_script_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_script_path)
        parent_directory = os.path.dirname(current_directory)
        file_path = os.path.join(parent_directory, "TestData/kgdata.npy")
        kappa, gamma, nside = read_glass_data(file_name=file_path)
        self.kappa = kappa
        self.gamma = gamma
        self.nside = nside

    def test_default_file_name(self):
        assert self.kappa is not None
        assert self.gamma is not None
        assert self.nside == 128
        with pytest.raises(ValueError):
            read_glass_data(file_name="xx")

    def test_generate_samples_from_glass(self):
        kappa_random_glass, gamma_random_glass = generate_samples_from_glass(
            self.kappa, self.gamma, 100
        )
        assert len(kappa_random_glass) == len(gamma_random_glass) == 100
        assert isinstance(kappa_random_glass, np.ndarray)
        assert isinstance(gamma_random_glass, np.ndarray)

    def test_skyarea_form_n(self):
        skyarea = skyarea_form_n(self.nside)
        assert skyarea == pytest.approx(0.20982341130279172, rel=1e-4)

        skyarea2 = skyarea_form_n(self.nside, deg2=False)
        assert skyarea2 == pytest.approx(2719311.41, rel=1e-2)

    def test_generate_maps_kmean_zero_using_halos(self):
        np.random.seed(41)
        kappa_random_glass, gamma_random_glass = generate_maps_kmean_zero_using_halos(
            samples_number_for_one_halos=50, renders_numbers=50
        )
        assert len(kappa_random_glass) == len(gamma_random_glass) == 50
        assert isinstance(kappa_random_glass, np.ndarray)
        assert isinstance(gamma_random_glass, np.ndarray)

    def test_generate_m_h_m_t_and_halos_plus_glass(self):
        kappa_random_glass, gamma_random_glass = generate_meanzero_halos_multiple_times(
            samples_number_for_one_halos=50,
            n_times=2,
            renders_numbers=2,
            skyarea=0.0001,
        )

        assert len(kappa_random_glass) == len(gamma_random_glass) == 4
        assert isinstance(kappa_random_glass, np.ndarray)
        assert isinstance(gamma_random_glass, np.ndarray)

        kappa_tot, gamma_tot = halos_plus_glass(
            self.kappa, self.gamma, kappa_random_glass, gamma_random_glass
        )
        assert len(kappa_tot) == len(gamma_tot) == 4
        assert isinstance(kappa_tot, (list, np.ndarray))
        assert isinstance(gamma_tot, (list, np.ndarray))

    def test_generate_m_h_m_t_for_small_area(self):
        np.random.seed(41)
        kappa_random_glass, gamma_random_glass = generate_meanzero_halos_multiple_times(
            samples_number_for_one_halos=5,
            n_times=2,
            renders_numbers=4,
            skyarea=0.0000001,
        )

        assert len(kappa_random_glass) == len(gamma_random_glass) == 8
        assert isinstance(kappa_random_glass, np.ndarray)
        assert isinstance(gamma_random_glass, np.ndarray)

        kappa_tot, gamma_tot = halos_plus_glass(
            self.kappa, self.gamma, kappa_random_glass, gamma_random_glass
        )
        assert len(kappa_tot) == len(gamma_tot) == 8
        assert isinstance(kappa_tot, (list, np.ndarray))
        assert isinstance(gamma_tot, (list, np.ndarray))

    def test_run_halos_without_kde(self):
        (
            kappa_run_halos_without_kde,
            gamma_run_halos_without_kde,
        ) = run_halos_without_kde(n_iterations=2, sky_area=0.00003, samples_number=5)
        assert (
            len(kappa_run_halos_without_kde) == len(gamma_run_halos_without_kde) == 10
        )
        assert isinstance(kappa_run_halos_without_kde, (list, np.ndarray))
        assert isinstance(gamma_run_halos_without_kde, (list, np.ndarray))

        (
            kappa_run_halos_without_kde2,
            gamma_run_halos_without_kde2,
        ) = run_halos_without_kde(
            n_iterations=2,
            sky_area=0.00003,
            samples_number=5,
            mass_sheet_correction=False,
        )
        assert (
            len(kappa_run_halos_without_kde2) == len(gamma_run_halos_without_kde2) == 10
        )

    def test_run_halos_without_kde_by_multiprocessing(self):
        (
            kappa_run_halos_without_kde_by_multiprocessing,
            gamma_run_halos_without_kde_by_multiprocessing,
        ) = run_halos_without_kde_by_multiprocessing(
            n_iterations=2, sky_area=0.00003, samples_number=5
        )
        assert (
            len(kappa_run_halos_without_kde_by_multiprocessing)
            == len(gamma_run_halos_without_kde_by_multiprocessing)
            == 10
        )
        assert isinstance(
            kappa_run_halos_without_kde_by_multiprocessing, (list, np.ndarray)
        )
        assert isinstance(
            gamma_run_halos_without_kde_by_multiprocessing, (list, np.ndarray)
        )

        (
            kappa2,
            gamma2,
        ) = run_halos_without_kde_by_multiprocessing(
            n_iterations=35, sky_area=0.00003, samples_number=2
        )
        assert isinstance(kappa2, (list, np.ndarray))
        assert isinstance(gamma2, (list, np.ndarray))

    def test_run_kappaext_gammaext_kde_by_multiprocessing(self):
        kappaext_gammaext_values_total = run_kappaext_gammaext_kde_by_multiprocessing(
            n_iterations=1, sky_area=0.0001, samples_number=1, z_max=0.3
        )
        for item in kappaext_gammaext_values_total:
            assert "zd" in item and isinstance(item["zd"], (float, int))
            assert "zs" in item and isinstance(item["zs"], (float, int))
            assert "kappa" in item and isinstance(item["kappa"], (float, int))
            assert "gamma" in item and isinstance(item["gamma"], (float, int))


def test_run_certain_redshift_lensext_kde_by_multiprocessing():
    result = run_certain_redshift_lensext_kde_by_multiprocessing()
    assert isinstance(result, list)


def test_run_certain_redshift_many_by_multiprocessing():
    k_g_values, lensinstance_values = run_certain_redshift_many_by_multiprocessing()
    assert isinstance(k_g_values, list)
    assert isinstance(lensinstance_values, list)


def test_run_total_kappa_by_multiprocessing():
    result = run_total_kappa_by_multiprocessing()
    assert isinstance(result, list)


def test_run_total_mass_by_multiprocessing():
    result = run_total_mass_by_multiprocessing()
    assert isinstance(result[0], float)


def test_convergence_mean_0():
    kappa_data = [1, 2, 3, 4, 5]
    adjusted_kappa_data = convergence_mean_0(kappa_data)
    assert adjusted_kappa_data == [-2.0, -1.0, 0.0, 1.0, 2.0]

    kappa_data2 = [1, 0, 2, 3, 4, 5, 0, 0, 0, 0]
    adjusted_kappa_data2 = convergence_mean_0(kappa_data2)
    assert adjusted_kappa_data2 == [-2.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0, 0, 0, 0]

    kappa_data3 = np.array([1, 2, 3, 4, 5])
    adjusted_kappa_data3 = convergence_mean_0(kappa_data3)
    assert isinstance(adjusted_kappa_data3, np.ndarray)


def test_worker_run_total_kappa_by_multiprocessing():
    iter_num = 1
    sky_area = 0.0001
    diff = 0.0000001
    num_points = 500
    diff_method = "square"
    m_min = 1.0e12
    m_max = 1.0e16
    z_max = 5.0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    result = worker_run_total_kappa_by_multiprocessing(
        iter_num,
        sky_area,
        diff,
        num_points,
        diff_method,
        m_min,
        m_max,
        z_max,
        cosmo,
    )

    assert isinstance(result, float)


def test_worker_run_total_mass_by_multiprocessing():
    iter_num = 1
    sky_area = 0.001
    m_min = 1e13
    m_max = 1e15
    z_max = 5.0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    total_mass = worker_run_total_mass_by_multiprocessing(
        iter_num, sky_area, m_min, m_max, z_max, cosmo
    )
    assert isinstance(total_mass, float)


def test_worker_certain_redshift_many():
    iter_num = 0
    sky_area = 0.0001
    m_min = 1.0e12
    m_max = 1.0e16
    z_max = 5.0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    samples_number = 1
    zs = 1.5
    zd = 1.0
    distributions, lensinstance = worker_certain_redshift_many(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        True,
        zs,
        zd,
    )
    distributions2, lensinstance2 = worker_certain_redshift_many(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        False,
        zs,
        zd,
    )
    assert isinstance(distributions, np.ndarray)
    assert isinstance(lensinstance, np.ndarray)
    assert isinstance(distributions2, np.ndarray)
    assert isinstance(lensinstance2, np.ndarray)


def test_worker_certain_redshift_lensext_kde():
    iter_num = 0
    sky_area = 0.0001
    m_min = 1.0e12
    m_max = 1.0e16
    z_max = 5.0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    samples_number = 1
    zs = 1.5
    zd = 1.0
    listmean = False
    sigma8 = 0.8
    omega_m = 0.3

    distributions = worker_certain_redshift_lensext_kde(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        True,
        zs,
        zd,
        listmean,
        sigma8,
        omega_m,
    )

    distributions2 = worker_certain_redshift_lensext_kde(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        False,
        zs,
        zd,
        listmean,
        sigma8,
        omega_m,
    )

    assert isinstance(distributions, np.ndarray)
    assert isinstance(distributions2, np.ndarray)


def test_worker_kappaext_gammaext_kde():
    iter_num = 0
    sky_area = 0.0001
    m_min = 1.0e15
    m_max = 1.0e16
    z_max = 0.3
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    samples_number = 1
    listmean = False
    output_format = "dict"

    distributions = worker_kappaext_gammaext_kde(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        True,
        listmean,
        output_format,
    )

    distributions2 = worker_kappaext_gammaext_kde(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        False,
        listmean,
        output_format,
    )

    assert isinstance(distributions, list)
    assert isinstance(distributions[0], dict)

    assert isinstance(distributions2, list)
    assert isinstance(distributions2[0], dict)


def test_worker_run_halos_without_kde():
    iter_num = 1
    sky_area = 0.0001
    m_min = 1.0e14
    m_max = 1.0e16
    z_max = 5.0
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    samples_number = 2
    listmean = False
    sigma8 = 0.8
    omega_m = 0.30

    nkappa, ngamma = worker_run_halos_without_kde(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        True,
        listmean,
        sigma8,
        omega_m,
    )

    nkappa2, ngamma2 = worker_run_halos_without_kde(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        False,
        listmean,
        sigma8,
        omega_m,
    )
    assert isinstance(nkappa, np.ndarray)
    assert len(nkappa) == samples_number
    assert isinstance(ngamma, np.ndarray)
    assert isinstance(nkappa2, np.ndarray)
    assert isinstance(ngamma2, np.ndarray)


def test_run_average_mass_by_multiprocessing():
    iter_num = 5
    sky_area = 0.0001
    m_min = 1.0e11
    m_max = 1.0e16
    z_max = 5.0

    average_masses = run_average_mass_by_multiprocessing(
        n_iterations=iter_num,
        sky_area=sky_area,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
    )

    assert isinstance(average_masses, np.ndarray)

    iter_num = 1

    average_masses_run = worker_run_average_mass_by_multiprocessing(
        iter_num=iter_num,
        sky_area=sky_area,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
    )

    assert isinstance(average_masses_run, np.ndarray)

    assert len(average_masses) == len(average_masses_run)
