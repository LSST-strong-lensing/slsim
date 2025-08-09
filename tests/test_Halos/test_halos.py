import numpy as np
from slsim.Halos.halos import (
    colossus_halo_mass_function,
    get_value_if_quantity,
    colossus_halo_mass_sampler,
    number_density_at_redshift,
    halo_mass_at_z,
    set_defaults_halos,
    dndz_to_redshifts,
    dv_dz_to_dn_dz,
    dndz_to_N,
    v_per_redshift,
    redshift_halos_array_from_comoving_density,
    redshift_mass_sheet_correction_array_from_comoving_density,
    number_density_for_massf,
    kappa_ext_for_each_sheet,
    expected_mass_at_redshift,
    colossus_halo_expected_mass_sampler,
    colossus_halo_expected_number_certain_bin,
    colossus_halo_expected_number,
    optimize_min_mass_based_on_number,
)

from astropy.cosmology import default_cosmology
from astropy.units.quantity import Quantity
from astropy import units
import pytest

cosmo = default_cosmology.get()


def test_colossus_halo_mass_function():
    m_200 = np.array([1e12, 1e13, 1e14])
    z = 0.5
    result = colossus_halo_mass_function(m_200, cosmo, z)
    assert result.shape == m_200.shape
    with pytest.raises(ValueError):
        colossus_halo_mass_function(m_200, cosmo, z, omega_m="wrong")


def test_returns_value_if_quantity():
    quantity = Quantity(5, unit="Msun")

    result = get_value_if_quantity(quantity)

    assert result == 5


def test_returns_array_of_size_when_all_parameters_valid():
    # Arrange
    m_min = 1e12
    m_max = 1e15
    resolution = 100
    z = 0.5
    size = 10

    result = colossus_halo_mass_sampler(
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        z=z,
        cosmology=cosmo,
        sigma8=0.81,
        ns=0.96,
        size=size,
    )

    assert len(result) == size
    assert np.all(result > m_min)
    assert np.all(result < m_max)


def test_halo_mass_at_z():
    mass = halo_mass_at_z(z=0.5, resolution=100)
    assert mass[0] > 10**10
    assert mass[0] < 10**16
    assert len(mass) == 1

    z_array = np.linspace(0, 1, 100)
    mass_list = halo_mass_at_z(z=z_array, m_min=1e12, m_max=1e16, resolution=100)
    flat_mass_array = np.concatenate(mass_list)
    assert np.all(flat_mass_array > 1e12)
    assert np.all(flat_mass_array < 1e16)
    assert len(mass_list) == 100

    z_list = [0, 1, 2]
    mass_list2 = halo_mass_at_z(z=z_list, resolution=100)
    assert len(mass_list2) == 3

    z_list3 = [np.nan, np.nan]
    mass_list3 = halo_mass_at_z(z=z_list3, resolution=100)
    assert len(mass_list3) == 2
    assert mass_list3 == [0, 0]


def test_number_density_at_redshift():
    z = 0.5
    CDF = number_density_at_redshift(z=z)
    assert len(CDF) == 1
    assert CDF[0] > 0

    CDF = number_density_at_redshift(z=np.array([np.nan]))
    assert len(CDF) == 1

    z2 = np.array([0, 10, 20])
    CDF2 = number_density_at_redshift(z=z2)
    assert len(CDF2) == 3
    assert CDF2[0] > CDF2[2]

    z3 = np.array([np.nan, np.nan])
    with pytest.warns(Warning, match=r".*data lost*"):
        CDF3 = number_density_at_redshift(z=z3)
    assert len(CDF3) == 2
    assert CDF3 == [0.0001, 0.0001]


def test_number_for_certain_mass():
    m = np.geomspace(1e12, 1e16, 200)
    massf = np.array([1] * 200)
    cdf = number_density_for_massf(massf, m)
    massf2 = 2 * m
    cdf2 = number_density_for_massf(massf2, m)
    s = (1e16 - 1e12) * (1e16 + 1e12)
    assert cdf == pytest.approx(1e16 - 1e12, rel=0.001)
    assert cdf2 == pytest.approx(s, rel=0.001)


def test_defaults_set():
    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults_halos()
    assert m_min == 1e12
    assert m_max == 1e14
    assert resolution == 100
    from astropy.cosmology import default_cosmology

    assert cosmology == default_cosmology.get()


def test_v_per_redshift():
    sky_area = 41252.96 * units.deg**2  # totoal sky
    z = np.linspace(0, 1, 100)
    v = v_per_redshift(redshift_list=z, cosmology=cosmo, sky_area=sky_area)
    totalv = np.trapz(v, z)
    v2 = cosmo.comoving_volume(1).to_value("Mpc3")
    sky_area_2 = 0.0 * units.deg**2
    z_2 = 1.0
    v_0 = v_per_redshift(redshift_list=z_2, cosmology=cosmo, sky_area=sky_area_2)
    assert len(v) == 100
    assert totalv == pytest.approx(v2, rel=0.001)
    assert v_0 == 0.0


def test_dndz_to_N():
    z = np.linspace(0, 1, 100)
    dndz = np.ones(100)
    N = dndz_to_N(dndz, z)

    dndz_0 = np.zeros(100)
    N_0 = dndz_to_N(dndz_0, z)

    assert isinstance(N, int)
    assert N == pytest.approx(1, rel=10.0)
    assert N_0 == 0


def test_dndz_to_redshifts():
    z = np.linspace(0, 2, 100)
    dndz = np.concatenate([np.array([0] * 50), np.linspace(0, 1, 50)])
    redshifts = dndz_to_redshifts(18, dndz, z)

    assert len(redshifts) == 18
    assert np.all(redshifts > 0.999)
    assert np.all(redshifts < 2.001)


def test_dv_dz_to_dn_dz():
    z = np.linspace(0, 1, 100)
    dV_dz = np.concatenate([np.array([0] * 50), np.linspace(0, 1, 50)])
    dn_dz = dv_dz_to_dn_dz(dV_dz, z)
    assert len(dn_dz) == 100
    assert dn_dz[0] == dn_dz[10] == 0
    assert dn_dz[60] > 0


def test_redshift_halos_array_from_comoving_density():
    redshift_list = [0.5, 1.0, 1.5]
    sky_area = 1 * units.deg**2
    cosmology = cosmo
    m_min = 1e12
    m_max = 1e15
    resolution = 100
    result = redshift_halos_array_from_comoving_density(
        redshift_list, sky_area, cosmology, m_min, m_max, resolution
    )
    assert isinstance(result, np.ndarray)
    assert len(result) > 1
    with pytest.warns(Warning, match=r".*default*"):
        redshift_halos_array_from_comoving_density(redshift_list, sky_area)
    with pytest.warns(Warning, match=r".*No Halos*"):
        redshift_list_small = [0.1, 0.2, 0.3]
        sky_area_small = 0.0000001 * units.deg**2
        m_min_no_h = 1e16
        m_max_no_h = 1e17
        redshift_halos_array_from_comoving_density(
            redshift_list_small, sky_area_small, cosmology, m_min_no_h, m_max_no_h
        )


def test_redshift_mass_sheet_correction_array():
    redshift_list = [0.1, 0.2]
    expected_result = np.array([0.025, 0.075, 0.125, 0.175])
    assert np.allclose(
        redshift_mass_sheet_correction_array_from_comoving_density(redshift_list),
        expected_result,
    )
    kappa = kappa_ext_for_each_sheet(
        redshift_list=redshift_mass_sheet_correction_array_from_comoving_density(
            redshift_list
        ),
        first_moment=[0.1, 0.2, 0.3, 0.4],
        sky_area=0.0000001 * units.deg**2,
        cosmology=cosmo,
    )
    kappa2 = kappa_ext_for_each_sheet(
        redshift_list=redshift_mass_sheet_correction_array_from_comoving_density(
            redshift_list
        ),
        first_moment=[0.1, 0.2, 0.3, 0.4],
        sky_area=0.0000001 * units.deg**2,
        cosmology=cosmo,
        z_sigma_crit_source=1.0,
    )
    assert len(kappa) == 4
    assert kappa[0] < 0
    assert len(kappa2) == 4
    assert kappa2[0] < 0
    assert kappa2[1] != kappa[1]


def test_returns_array_of_kappa_ext():
    redshift_list = [0.1, 0.2]
    first_moment = [0.1, 0.2]
    sky_area = Quantity(value=0.1, unit="deg2")
    cosmology = cosmo

    result = kappa_ext_for_each_sheet(redshift_list, first_moment, sky_area, cosmology)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(redshift_list)


def test_mass_first_moment_at_redshift():
    z = [0.025, 0.075]
    sky_area = 0.001 * units.deg**2
    m_min = 1e11
    m_max = 1e13
    resolution = 50
    cosmology = cosmo
    sigma8 = 0.8
    ns = 0.95
    omega_m = 0.25

    result = expected_mass_at_redshift(
        z, sky_area, m_min, m_max, resolution, cosmology, sigma8, ns, omega_m
    )

    assert len(result) == 2


def test_returns_float_with_valid_input_values():
    m_min = 1e12
    m_max = 1e15
    resolution = 100
    z = 0.5
    cosmology = cosmo

    result = colossus_halo_expected_mass_sampler(
        m_min,
        m_max,
        resolution,
        z,
        cosmology,
        sigma8=0.81,
        ns=0.96,
    )

    assert isinstance(result, float)


def test_colossus_halo_number_first_moment_certain_bin():
    z_c = 1.0
    dz = 0.1
    sky_area = 0.0001 * units.deg**2
    m_min = 1e11
    m_max = 1e15
    resolution = 100
    cosmology = cosmo

    result = colossus_halo_expected_number_certain_bin(
        z_c,
        dz,
        sky_area,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )

    assert isinstance(result, float)
    assert result > 0

    dz2 = 0.3
    result2 = colossus_halo_expected_number_certain_bin(
        z_c,
        dz2,
        sky_area,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )
    assert result2 > result

    result3 = colossus_halo_expected_number_certain_bin(
        z_c=0.0001,
        dz=0.00005,
        sky_area=0.0001 * units.deg**2,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )
    assert result3 == pytest.approx(expected=0.0, abs=1e-2)


def test_colossus_halo_expected_number():
    z = 5.0
    sky_area = 0.0001 * units.deg**2
    m_min = 1e11
    m_max = 1e15
    resolution = 100
    cosmology = cosmo

    result = colossus_halo_expected_number(
        zmax=z,
        sky_area=sky_area,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )

    assert isinstance(result, float)
    assert result > 0

    result_other = colossus_halo_expected_number_certain_bin(
        z_c=2.5,
        dz=5.0,
        sky_area=sky_area,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )
    assert result_other == pytest.approx(result, abs=1e-2)


def test_optimize_min_mass_based_on_number():
    z = 5.0
    sky_area = 0.0001 * units.deg**2
    m_max = 1e15
    resolution = 100
    cosmology = cosmo
    number = 100
    result = optimize_min_mass_based_on_number(
        target_n_halos=number,
        zmax=z,
        sky_area=sky_area,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )
    assert isinstance(result, float)
    assert result < m_max
    assert result >= 1e9

    result2 = optimize_min_mass_based_on_number(
        target_n_halos=100,
        zmax=0.00001,
        sky_area=sky_area,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=0.81,
        ns=0.96,
    )
    assert result2 == 1e9
