from slsim.Halos.halos_statistics import HalosStatistics
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table


@pytest.fixture
def setup_halos_hs():
    z = [0.5, 0.6, 0.7]

    mass = [2058751081954.2866, 1320146153348.6448, 850000000000.0]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
    )
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    HS = HalosStatistics(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        samples_number=3,
        mass_sheet=True,
    )
    return HS


@pytest.fixture
def setup_no_halos():
    z = [np.nan]

    mass = [0]

    halos = Table([z, mass], names=("z", "mass"))

    z_correction = [0.5]
    kappa_ext_correction = [-0.1]
    mass_sheet_correction = Table(
        [z_correction, kappa_ext_correction], names=("z", "kappa")
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return HalosStatistics(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=0.0001,
        cosmo=cosmo,
        samples_number=2,
        mass_sheet=True,
    )


def test_get_kappaext_gammaext_distib_zdzs(setup_halos_hs):
    hs = setup_halos_hs
    zd = 0.5
    zs = 1.0
    kappa_gamma_distribution = hs.get_kappaext_gammaext_distib_zdzs(zd, zs)

    assert kappa_gamma_distribution.shape == (hs.samples_number, 2)

    k_g_distribution = hs.get_kappaext_gammaext_distib_zdzs(zd, zs, listmean=True)
    assert k_g_distribution.shape == (hs.samples_number, 2)
    assert np.mean(k_g_distribution[:, 0]) == pytest.approx(0, abs=1e-6)


def test_generate_distributions_0to5(setup_halos_hs):
    hs = setup_halos_hs
    distributions = hs.generate_distributions_0to5()

    assert isinstance(distributions, list)
    for entry in distributions:
        assert "zd" in entry and isinstance(entry["zd"], float)
        assert "zs" in entry and isinstance(entry["zs"], float)
        assert "kappa" in entry and isinstance(entry["kappa"], float)
        assert "gamma" in entry and isinstance(entry["gamma"], float)

    distributions2 = hs.generate_distributions_0to5(output_format="vector")
    assert isinstance(distributions2, list)

    pytest.raises(
        ValueError, hs.generate_distributions_0to5, output_format="wrong_format"
    )


def test_compute_various_kappa_gamma_values(setup_halos_hs):
    hs = setup_halos_hs
    zd = 0.5
    zs = 1.0
    (
        kappa_od,
        kappa_os,
        gamma_od1,
        gamma_od2,
        gamma_os1,
        gamma_os2,
        kappa_ds,
        gamma_ds1,
        gamma_ds2,
    ) = hs.compute_various_kappa_gamma_values(zd, zs)
    assert isinstance(kappa_od, float)
    assert isinstance(kappa_os, float)
    assert isinstance(gamma_od1, float)
    assert isinstance(gamma_od2, float)
    assert isinstance(gamma_os1, float)
    assert isinstance(gamma_os2, float)
    assert isinstance(kappa_ds, float)
    assert isinstance(gamma_ds1, float)
    assert isinstance(gamma_ds2, float)

    kappa_gamma_distribution, lens_instance = hs.get_all_pars_distib(zd, zs)
    assert isinstance(kappa_gamma_distribution, np.ndarray)
    assert isinstance(lens_instance, np.ndarray)

    all_kappa_dicts = hs.compute_kappa_in_bins()
    assert isinstance(all_kappa_dicts, list)


def test_total_mass(setup_halos_hs, setup_no_halos):
    hs = setup_halos_hs
    mass = hs.total_halo_mass()
    assert isinstance(mass, float)
    assert mass == (2058751081954.2866 + 1320146153348.6448 + 850000000000.0)

    hl2 = setup_no_halos
    mass2 = hl2.total_halo_mass()
    assert mass2 == 0


def test_total_critical_mass(setup_halos_hs, setup_no_halos):
    hl = setup_halos_hs
    mass1 = hl.total_critical_mass(method="differential_comoving_volume")
    mass2 = hl.total_critical_mass(method="comoving_volume")
    assert isinstance(mass1, float)
    assert isinstance(mass2, float)
    ratio = mass1 / mass2
    assert ratio == pytest.approx(1, rel=0.01)


def test_get_kappa_gamma_distib_without_multiprocessing(setup_halos_hs, setup_no_halos):
    hl = setup_halos_hs
    results = hl.get_kappa_gamma_distib_without_multiprocessing()
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 3  # kappa, gamma1, gamma2

    results = hl.get_kappa_gamma_distib_without_multiprocessing(gamma_tot=True)
    assert results.shape[0] == hl.samples_number
    assert results.shape[1] == 2  # kappa, gamma

    results2 = hl.get_kappa_gamma_distib_without_multiprocessing(
        gamma_tot=True, listmean=True
    )
    assert results2.shape[0] == hl.samples_number

    results2 = hl.get_kappa_gamma_distib_without_multiprocessing(
        gamma_tot=False, listmean=True
    )
    assert results2.shape[0] == hl.samples_number

    hl2 = setup_no_halos
    results2 = hl2.get_kappa_gamma_distib_without_multiprocessing(gamma_tot=True)
    assert results2.shape[0] == hl2.samples_number
    assert results2.shape[1] == 2  # kappa, gamma


def test_kappa_divergence(setup_halos_hs):
    hl = setup_halos_hs
    kappa_div = hl.kappa_divergence()
    assert isinstance(kappa_div, float)
