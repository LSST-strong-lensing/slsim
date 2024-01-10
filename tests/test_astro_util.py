import numpy as np
from numpy import testing as npt
import pytest
from astropy import constants as const
from astropy import units as u
from slsim.Util.astro_util import (
    spin_to_isco,
    calculate_eddington_luminosity,
    eddington_ratio_to_accretion_rate,
    calculate_gravitational_radius,
    convert_black_hole_mass_exponent_to_mass,
    thin_disk_temperature_profile,
    planck_law,
    planck_law_derivative,
    create_radial_map,
    create_phi_map,
    calculate_time_delays_on_disk,
)


def test_spin_to_isco():
    # Check all special cases of
    # spin = 0, the Schwarzschild black hole
    # spin = 1, maximum prograde Kerr black hole
    # spin = -1, maximum retrograde Kerr black hole
    assert spin_to_isco(0.0) == 6.0
    assert spin_to_isco(-1.0) == 9.0
    assert spin_to_isco(1.0) == 1.0
    # test errors
    with pytest.raises(ValueError):
        spin_to_isco(2.0)
        spin_to_isco(-100.0)


def test_calculate_eddington_luminosity():
    # Check some cases
    test_value_1 = (
        4 * np.pi * const.G * const.m_p * const.c * const.M_sun / const.sigma_T
    )
    npt.assert_approx_equal(calculate_eddington_luminosity(0).value, test_value_1.value)

    test_value_2 = (
        4
        * np.pi
        * const.G
        * const.m_p
        * 100000000
        * const.c
        * const.M_sun
        / const.sigma_T
    )
    npt.assert_approx_equal(calculate_eddington_luminosity(8).value, test_value_2.value)
    assert calculate_eddington_luminosity(8).unit == test_value_2.unit


def test_eddington_ratio_to_accretion_rate():
    # Test zero efficiency error
    with pytest.raises(ValueError):
        eddington_ratio_to_accretion_rate(8.0, 0.1, efficiency=0)

    # Test zero eddington ratio
    assert eddington_ratio_to_accretion_rate(8.0, 0.0) == 0

    # Test typical value
    expected_value = calculate_eddington_luminosity(8.0) * 0.1 / (0.1 * const.c**2)

    npt.assert_approx_equal(
        eddington_ratio_to_accretion_rate(8.0, 0.1).value, expected_value.value
    )
    assert eddington_ratio_to_accretion_rate(8.0, 0.1).unit == expected_value.unit


def test_calculate_gravitational_radius():
    # Test expected scaling
    npt.assert_approx_equal(
        calculate_gravitational_radius(7.0).value * 10,
        calculate_gravitational_radius(8.0).value,
    )
    # Test low value
    expected_value = const.G * const.M_sun / const.c**2
    npt.assert_approx_equal(
        calculate_gravitational_radius(0.0).value, expected_value.value
    )


def test_convert_black_hole_mass_exponent_to_mass():
    expected_value = 10**8.0 * const.M_sun
    npt.assert_approx_equal(
        convert_black_hole_mass_exponent_to_mass(8.0).value, expected_value.value
    )


def test_thin_disk_temperature_profile():
    # define points to test temperature for various spins
    # reminder: important spins are [-1, 0, 1], which have R_in at [9, 6, 1]
    # at very high radius, should have negligable temp
    r_points = np.asarray([0, 2, 5, 7, 8, 10, 1e30])
    profile_1 = thin_disk_temperature_profile(r_points, -1, 8.0, 0.1)
    assert profile_1[0].value == 0
    assert profile_1[1] == profile_1[0]
    assert profile_1[2] == profile_1[0]
    assert profile_1[3] == profile_1[0]
    assert profile_1[4] == profile_1[0]
    assert profile_1[5] > profile_1[4]
    # Check profile has converged to a small temperature and units are correct
    assert profile_1[6].value < 1
    assert profile_1[6].decompose().unit == u.K

    profile_2 = thin_disk_temperature_profile(r_points, 0, 8.0, 0.1)
    assert profile_2[2] == profile_1[2]
    assert profile_2[3] > profile_2[2]
    assert profile_2[-1].value < 1

    profile_3 = thin_disk_temperature_profile(r_points, 1, 8.0, 0.1)
    assert profile_3[1] > profile_3[0]
    assert profile_3[-1].value < 1

    # Check we can get a 0 profile by setting accretion rate to 0
    profile_zero = thin_disk_temperature_profile(r_points, 1, 8.0, 0.0)
    assert np.sum(profile_zero) == 0


def test_planck_law():
    # Test zero temperature means zero emission
    assert planck_law(0 * u.K, 100) == 0
    assert planck_law(0, 1000) == planck_law(0, 50000)
    # Test increasing temperature always increases spectral radiance
    radiances = planck_law(np.linspace(100, 500, 20), 300)
    delta_radiances = radiances[1:] - radiances[:-1]
    assert all(delta_radiances > 0)
    # Test peak shifts with temperature
    wavelengths = np.linspace(100, 5000, 500)
    spectral_profile_low_temp = planck_law(500, wavelengths)
    spectral_profile_med_temp = planck_law(2500, wavelengths)
    spectral_profile_high_temp = planck_law(5000, wavelengths)
    assert np.argmax(spectral_profile_low_temp) > np.argmax(spectral_profile_med_temp)
    assert np.argmax(spectral_profile_med_temp) > np.argmax(spectral_profile_high_temp)

    # Test choice of units does not matter on input (if units are used correctly)
    npt.assert_approx_equal(
        planck_law(500, 200).decompose().value,
        planck_law(500, 200 * 1e-6 * u.mm).decompose().value,
    )


def test_planck_law_derivative():
    # Test against a smaller delta_temperature manually
    expected_value = (planck_law(10000 + 1e-8, 500) - planck_law(10000, 500)) / 1e-8
    npt.assert_approx_equal(
        planck_law_derivative(10000, 500).value, expected_value.value, 4
    )


def test_create_radial_map():
    # Test shape is as expected
    radial_map = create_radial_map(100, 100, 0)
    assert np.shape(radial_map) == (200, 200)
    # Test corner values
    assert radial_map[0, 0] == 100 * 2**0.5
    assert radial_map[0, -1] == radial_map[0, 0]
    assert radial_map[-1, -1] == radial_map[0, 0]
    assert radial_map[-1, 0] == radial_map[0, 0]


def test_create_phi_map():
    # Test shape matches radial_map
    phi_map = create_phi_map(100, 100, 45)
    radial_map = create_radial_map(100, 100, 45)
    assert np.shape(phi_map) == np.shape(radial_map)

    # Test phi = 0 points towards the observer, the bottom side of the map
    assert phi_map[100, 0] < 1e-2

    # Test phi values rotate counter-clockwise
    assert phi_map[110, 0] > phi_map[100, 0]
    assert phi_map[-1, -1] > phi_map[-1, 0]


def test_calculate_time_delays_on_disk():
    # Test all time delays are tiny for a very low black hole mass exponent
    time_delay_map = calculate_time_delays_on_disk(100, 100, 40, 2, 10)
    # the average time delay is the sum of all time delays divided by the number of points.
    assert np.sum(time_delay_map) / (200**2) < (1 * u.s)

    # Test maximum time delay points away from observer
    assert np.argmax(time_delay_map[100, :]) == 199

    # Test left- right-side symmetry
    npt.assert_approx_equal(time_delay_map[0, 25].value, time_delay_map[-1, 25].value)
    npt.assert_approx_equal(time_delay_map[0, 100].value, time_delay_map[-1, 100].value)
    npt.assert_approx_equal(time_delay_map[0, 180].value, time_delay_map[-1, 180].value)
