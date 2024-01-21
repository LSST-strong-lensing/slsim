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
    calculate_geometric_contribution_to_lamppost_model,
    calculate_dt_dlx,
    calculate_mean_time_lag,
    calculate_accretion_disk_response_function,
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

    # Test phi = 0 points towards the compressed side of the map
    assert radial_map[0, 100] > 100
    assert phi_map[0, 100] < 1e-2

    # Test phi values rotate counter-clockwise
    assert phi_map[0, 110] > phi_map[0, 100]


def test_calculate_time_delays_on_disk():
    radial_map = create_radial_map(100, 100, 45)
    phi_map = create_phi_map(100, 100, 45)
    # Test all time delays are tiny for a very low black hole mass exponent
    time_delay_map = calculate_time_delays_on_disk(radial_map, phi_map, 40, 10)
    low_mass_time_delay_map = (
        time_delay_map * calculate_gravitational_radius(2.0) / const.c
    )
    # the average time delay is the sum of all time delays divided by the number of points.
    assert np.sum(low_mass_time_delay_map) / (200**2) < (1 * u.s)

    # Test maximum time delay points away from observer
    assert np.argmax(time_delay_map[:, 100]) == 199

    # Test left- right-side symmetry
    npt.assert_approx_equal(time_delay_map[25, 0], time_delay_map[25, -1])
    npt.assert_approx_equal(time_delay_map[100, 0], time_delay_map[100, -1])
    npt.assert_approx_equal(time_delay_map[180, 0], time_delay_map[180, -1])

    # Test single value for a face on disk
    # (light travels back and forth for total of 20 R_g)
    radial_map = np.array([[0]])
    phi_map = np.array([[0]])
    time_delay_map = calculate_time_delays_on_disk(radial_map, phi_map, 0, 10)
    npt.assert_equal(time_delay_map, np.array([[20]]))


def test_calculate_geometric_contribution_to_lamppost_model():
    radial_map = create_radial_map(100, 100, 45)
    # Test that 0 corona height doesn't cause error
    values_low = calculate_geometric_contribution_to_lamppost_model(radial_map, 0)
    # Test that a very high X-ray source leads to large suppression
    values_high = calculate_geometric_contribution_to_lamppost_model(radial_map, 1e8)
    assert np.sum(values_high) < 1e-5
    # Test the close corona has a greater impact than far corona
    assert np.sum(values_low) > np.sum(values_high)


def test_calculate_dt_dlx():
    radial_map = create_radial_map(100, 100, 45)
    temperature_map = thin_disk_temperature_profile(radial_map, 0.0, 8.0, 0.1)
    dt_dlx_map = calculate_dt_dlx(radial_map, temperature_map, 10)
    # Test that the greatest dt_dlx occurs near the center of the disk
    # (the center of the disk is between points (99, 99) and (100, 100)
    # additionally, the ISCO leaves a dark region for R < 6 R_g
    assert abs(np.argmax(dt_dlx_map[:, 100]) - 100) <= 7
    assert abs(np.argmax(dt_dlx_map[100, :]) - 100) <= 7
    # Test dt_dlx is suppressed for extremely distant X-ray source
    # this calculates the average fractional temperature change over the disk
    dt_dlx_map = calculate_dt_dlx(radial_map, temperature_map, 1e8)
    mask = temperature_map > 0
    assert (
        np.sum(
            np.nan_to_num(dt_dlx_map / temperature_map * mask) / np.size(radial_map)
        ).value
        < 1e-10
    )


def test_calculate_mean_time_lag():
    # Test a simple case explicitly
    response_function = [0, 1, 2, 3, 4]
    expected_value = (0 + 1 + 4 + 9 + 16) / (0 + 1 + 2 + 3 + 4)
    assert calculate_mean_time_lag(response_function) == expected_value


def test_calculate_accretion_disk_response_function():
    # Test that manual construction of response function with previously
    # tested functions works exactly the same as this function
    r_out = 100
    r_resolution = 100
    inclination_angle = 45.0
    rest_frame_wavelength_in_nanometers = 1000.0
    black_hole_mass_exponent = 8.0
    black_hole_spin = 0.0
    corona_height = 10.0
    eddington_ratio = 0.1

    radial_map = create_radial_map(r_out, r_resolution, inclination_angle)
    phi_map = create_phi_map(r_out, r_resolution, inclination_angle)
    temperature_map = thin_disk_temperature_profile(
        radial_map, black_hole_spin, black_hole_mass_exponent, eddington_ratio
    )
    temperature_map *= radial_map < r_out
    db_dt_map = planck_law_derivative(
        temperature_map, rest_frame_wavelength_in_nanometers
    )

    dt_dlx_map = calculate_dt_dlx(radial_map, temperature_map, corona_height)
    weighting_factors = np.nan_to_num(db_dt_map * dt_dlx_map)
    time_delay_map = calculate_time_delays_on_disk(
        radial_map, phi_map, inclination_angle, corona_height
    )
    response_function_manual = np.histogram(
        time_delay_map,
        range=(0, np.max(time_delay_map) + 1),
        bins=int(np.max(time_delay_map) + 1),
        weights=weighting_factors,
        density=True,
    )[0]
    response_function_manual /= np.nansum(response_function_manual)

    response_function = calculate_accretion_disk_response_function(
        r_out,
        r_resolution,
        inclination_angle,
        rest_frame_wavelength_in_nanometers,
        black_hole_mass_exponent,
        black_hole_spin,
        corona_height,
        eddington_ratio,
    )
    npt.assert_array_almost_equal(
        response_function.value, response_function_manual.value, 5
    )

    # Test that an inclined disk produces a "skewed" response function (e.g. it peaks earlier)
    inclination_angle_face_on = 0.0
    response_function_face_on = calculate_accretion_disk_response_function(
        r_out,
        r_resolution,
        inclination_angle_face_on,
        rest_frame_wavelength_in_nanometers,
        black_hole_mass_exponent,
        black_hole_spin,
        corona_height,
        eddington_ratio,
    )
    peak_response_face_on = np.argmax(response_function_face_on)
    peak_response_inclined = np.argmax(response_function)
    assert peak_response_inclined < peak_response_face_on

    # Test that inclination does not change the mean response when H_{L_x} = 0
    # Note I add 0.5 within the function to avoid singularities when the corona
    # is placed directly on the disk. However there is no contribution with the dark ISCO.
    corona_height = -0.5
    response_function_face_on = calculate_accretion_disk_response_function(
        r_out,
        r_resolution,
        inclination_angle_face_on,
        rest_frame_wavelength_in_nanometers,
        black_hole_mass_exponent,
        black_hole_spin,
        corona_height,
        eddington_ratio,
    )
    response_function_inclined = calculate_accretion_disk_response_function(
        r_out,
        r_resolution,
        inclination_angle,
        rest_frame_wavelength_in_nanometers,
        black_hole_mass_exponent,
        black_hole_spin,
        corona_height,
        eddington_ratio,
    )

    mean_tau_face_on = calculate_mean_time_lag(response_function_face_on)
    mean_tau_inclined = calculate_mean_time_lag(response_function_inclined)

    npt.assert_equal(mean_tau_face_on, mean_tau_inclined)

    # Test that longer wavelengths produce broader response functions than short
    # wavelengths (e.g. they peak later and have longer mean time lags)
    corona_height = 10
    shorter_wavelength = 200
    longer_wavelength = 2000
    response_function_shorter_wavelength = calculate_accretion_disk_response_function(
        r_out,
        r_resolution,
        inclination_angle,
        shorter_wavelength,
        black_hole_mass_exponent,
        black_hole_spin,
        corona_height,
        eddington_ratio,
    )
    response_function_longer_wavelength = calculate_accretion_disk_response_function(
        r_out,
        r_resolution,
        inclination_angle,
        longer_wavelength,
        black_hole_mass_exponent,
        black_hole_spin,
        corona_height,
        eddington_ratio,
    )
    mean_tau_shorter_wavelength = calculate_mean_time_lag(
        response_function_shorter_wavelength
    )
    mean_tau_longer_wavelength = calculate_mean_time_lag(
        response_function_longer_wavelength
    )
    peak_response_longer_wavelength = np.argmax(response_function_longer_wavelength)
    peak_response_shorter_wavelength = np.argmax(response_function_shorter_wavelength)

    assert peak_response_shorter_wavelength < peak_response_longer_wavelength
    assert mean_tau_shorter_wavelength < mean_tau_longer_wavelength
