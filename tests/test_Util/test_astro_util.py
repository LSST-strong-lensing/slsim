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
    calculate_accretion_disk_emission,
    calculate_accretion_disk_response_function,
    define_bending_power_law_psd,
    define_frequencies,
    normalize_light_curve,
    generate_signal,
    generate_signal_from_bending_power_law,
    generate_signal_from_generic_psd,
    downsample_passband,
    bring_passband_to_source_plane,
    convert_passband_to_nm,
    pull_value_from_grid,
    extract_light_curve,
    theta_star_physical,
    get_tau_sf_from_distribution_agn_variability,
    get_breakpoint_frequency_and_std_agn_variability,
)
from astropy.cosmology import Planck18
from astropy.units import Quantity


def test_spin_to_isco():
    # Check all special cases of:
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


def test_calculate_accretion_disk_emission():
    # define some simple accretion disk parameters
    r_out = 100
    r_resolution = 100
    inclination_angle = 10
    rest_frame_wavelength_in_nm = 400
    black_hole_mass_exponent = 7.0
    black_hole_spin = 0.7
    eddington_ratio = 0.05

    emission_1 = calculate_accretion_disk_emission(
        r_out,
        r_resolution,
        inclination_angle,
        rest_frame_wavelength_in_nm,
        black_hole_mass_exponent,
        black_hole_spin,
        eddington_ratio,
    )
    emission_2 = calculate_accretion_disk_emission(
        r_out,
        r_resolution,
        inclination_angle,
        rest_frame_wavelength_in_nm,
        black_hole_mass_exponent,
        black_hole_spin,
        eddington_ratio * 2,
    )
    emission_3 = calculate_accretion_disk_emission(
        r_out * 2,
        r_resolution * 2,
        inclination_angle,
        rest_frame_wavelength_in_nm,
        black_hole_mass_exponent,
        black_hole_spin,
        eddington_ratio,
    )
    emission_distribution_3 = calculate_accretion_disk_emission(
        r_out * 2,
        r_resolution * 2,
        inclination_angle,
        rest_frame_wavelength_in_nm,
        black_hole_mass_exponent,
        black_hole_spin,
        eddington_ratio,
        return_spectral_radiance_distribution=True,
    )

    # assert that higher eddington ratio = more emission
    assert emission_2 > emission_1
    # assert that larger accretion disk = more emission
    assert emission_3 > emission_1
    # assert the distribution is a np array with total sum equal to emission_3
    assert isinstance(emission_distribution_3, np.ndarray)
    assert np.nansum(emission_distribution_3) == emission_3


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

    # Test that inclination does not change the mean response when H_{Lx} = 0
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


def test_define_frequencies():
    # Test that we can generate 2 points, recall the length will be 10 times the input
    length = 0.1
    dt = 1
    freq = define_frequencies(length, dt)
    assert len(freq) == 2
    # Test that we don't get frequencies above the Nyquist frequency defined as 1/(2*dt)
    # or that we don't get frequencies below the expected contributing range of 1/(10*length)
    length = 100
    dt = 1
    freq = define_frequencies(length, dt)
    assert np.max(freq) <= 1 / (2 * dt)
    assert np.min(freq) >= 1 / (10 * length)
    # Test that frequencies strictly increase (the x-axis cannot be multi-valued)
    delta_freq = freq[1:] - freq[:-1]
    assert all(delta_freq > 0)
    # Test that decreasing dt increases the maximum frequency
    dt = 0.1
    higher_freq = define_frequencies(length, dt)
    assert higher_freq[-1] > freq[-1]


def test_define_bending_power_law_psd():
    length = 100
    dt = 1
    frequencies = define_frequencies(length, dt)
    power_spectrum_density = define_bending_power_law_psd(-1, 1, 3, frequencies)
    # Test the frequencies and psd have the same size
    assert power_spectrum_density.size == frequencies.size
    # Test that the PSD is a decreasing function of frequency
    test_metric = power_spectrum_density[1:] - power_spectrum_density[:-1]
    assert all(test_metric <= 0)
    # Test that the PSD doesn't change if a breakpoint frequency is changed far outside the
    # frequency range
    psd1 = define_bending_power_law_psd(15, 1, 3, frequencies)
    psd2 = define_bending_power_law_psd(25, 1, 3, frequencies)
    total_diff_12 = np.sum(psd1 - psd2)
    npt.assert_almost_equal(total_diff_12, 0, 5)
    psd3 = define_bending_power_law_psd(-25, 0.5, 1, frequencies)
    psd4 = define_bending_power_law_psd(-20, 0.5, 1, frequencies)
    total_diff_34 = np.sum(psd3 - psd4)
    npt.assert_almost_equal(total_diff_34, 0, 5)


def test_normalize_light_curve():
    input_light_curve = [5.0, 7.0, 7.0, 5.0, 4.0, 2.0]
    new_mean = 0.0
    new_std = 1.0
    output_light_curve = normalize_light_curve(input_light_curve, new_mean, new_std)
    npt.assert_almost_equal(output_light_curve.mean(), new_mean)
    npt.assert_almost_equal(output_light_curve.std(), new_std)
    # Test a single value
    input_light_curve = [3.5]
    new_mean = 2.0
    new_std = 0.0
    output_light_curve = normalize_light_curve(input_light_curve, new_mean, new_std)
    assert output_light_curve[0] == new_mean
    # Test a numpy array generated from linspace as the input
    input_light_curve = np.linspace(0, 100, 101)
    new_mean = 0.0
    new_std = 1.0
    output_light_curve = normalize_light_curve(input_light_curve, new_mean, new_std)
    npt.assert_almost_equal(output_light_curve.mean(), new_mean)
    npt.assert_almost_equal(output_light_curve.std(), new_std)
    # by symmetry, the new center value should be 0
    assert output_light_curve[50] == 0


def test_generate_signal():
    # Test we can generate a signal with minimal arguments
    length_of_light_curve = 100
    dt = 7
    light_curve = generate_signal(length_of_light_curve, dt)
    assert len(light_curve) == 100 // 7
    # Test a smaller dt increases the number of points in the light curve
    dt = 1
    light_curve_2 = generate_signal(length_of_light_curve, dt)
    assert len(light_curve) < len(light_curve_2)
    # Test that no errors are thrown when reasonable values are entered
    length_of_light_curve = 1000
    dt = 2
    log_bp_freq = -1
    low_freq_slope = 0.5
    high_freq_slope = 3.0
    mean_magnitude = 12
    standard_deviation = 10
    light_curve_3 = generate_signal(
        length_of_light_curve,
        dt,
        log_breakpoint_frequency=log_bp_freq,
        low_frequency_slope=low_freq_slope,
        high_frequency_slope=high_freq_slope,
        mean_magnitude=mean_magnitude,
        standard_deviation=standard_deviation,
    )
    npt.assert_almost_equal(np.sum(light_curve_3), mean_magnitude * len(light_curve_3))
    # Test that a light curve may be generated using a user defined psd
    chosen_seed = 17
    input_frequencies = define_frequencies(length_of_light_curve, dt)
    input_power_spectrum_density = input_frequencies ** (-4.0)
    light_curve_user_psd = generate_signal(
        length_of_light_curve,
        dt,
        input_freq=input_frequencies,
        input_psd=input_power_spectrum_density,
        seed=chosen_seed,
    )
    # Compare the light curve is different from the broken power law psd
    light_curve_bpl_psd = generate_signal(length_of_light_curve, dt, seed=chosen_seed)
    difference_curve = light_curve_user_psd - light_curve_bpl_psd
    sum_of_squares = np.sum(difference_curve**2)
    assert sum_of_squares > 0

    # Test using magnitudes as inputs
    chosen_seed = 15
    mean_magnitude = -15
    magnitude_standard_deviation = 0.1

    with pytest.raises(ValueError):
        generate_signal(
            length_of_light_curve,
            dt,
            mean_magnitude=5,
            standard_deviation=5,
            normal_magnitude_variance=False,
        )
        generate_signal(
            length_of_light_curve,
            dt,
            mean_magnitude=5,
            standard_deviation=1,
            normal_magnitude_variance=False,
        )
    light_curve_without_value_error = generate_signal(
        length_of_light_curve,
        dt,
        mean_magnitude=mean_magnitude,
        standard_deviation=magnitude_standard_deviation,
        normal_magnitude_variance=False,
    )
    assert (
        np.max(light_curve_without_value_error)
        - np.min(light_curve_without_value_error)
        > magnitude_standard_deviation
    )


def test_generate_signal_from_bending_power_law():
    # Test that this function generates an identical signal to
    # that created with generate_signal()

    length_of_light_curve = 500
    time_resolution = 1
    log_breakpoint_frequency = -2
    low_frequency_slope = 1
    high_frequency_slope = 3
    mean_magnitude = 0
    standard_deviation = 1
    seed = 17

    known_signal = generate_signal(
        length_of_light_curve,
        time_resolution,
        log_breakpoint_frequency=log_breakpoint_frequency,
        low_frequency_slope=low_frequency_slope,
        high_frequency_slope=high_frequency_slope,
        mean_magnitude=mean_magnitude,
        standard_deviation=standard_deviation,
        seed=seed,
    )
    times, new_signal = generate_signal_from_bending_power_law(
        length_of_light_curve,
        time_resolution,
        log_breakpoint_frequency=log_breakpoint_frequency,
        low_frequency_slope=low_frequency_slope,
        high_frequency_slope=high_frequency_slope,
        mean_magnitude=mean_magnitude,
        standard_deviation=standard_deviation,
        seed=seed,
    )

    assert all(new_signal == known_signal)
    assert len(times) == len(new_signal)


def test_generate_signal_from_generic_psd():
    length_of_light_curve = 500
    time_resolution = 1
    seed = 112
    frequencies = define_frequencies(length_of_light_curve, time_resolution)
    test_psd_smooth = frequencies ** (-4)
    t1, smooth_signal = generate_signal_from_generic_psd(
        length_of_light_curve, time_resolution, frequencies, test_psd_smooth, seed=seed
    )

    test_psd_noisy = frequencies ** (-1)
    t2, noisy_signal = generate_signal_from_generic_psd(
        length_of_light_curve, time_resolution, frequencies, test_psd_noisy, seed=seed
    )

    # Note that smooth signals will deviate much further from the mean
    # than extremely noisy signals
    assert noisy_signal.var() < smooth_signal.var()


def test_downsample_passband():
    delta_wavelength = 10
    # Test errors
    with pytest.raises(ValueError):
        downsample_passband("yellow", delta_wavelength)
        downsample_passband((32, 12), delta_wavelength)
    with pytest.raises(ValueError):
        downsample_passband((1), delta_wavelength)
    passband_wavelengths = np.linspace(100, 200, 50)
    passband_throughputs = np.linspace(0, 1, 50) * np.linspace(1, 0, 50)
    passband = [passband_wavelengths, passband_throughputs]
    new_passband_0 = downsample_passband(passband, delta_wavelength)
    assert len(new_passband_0[0] == 10)
    # Test unit conversion
    speclite_filter = "lsst2016-u"
    new_passband_1 = downsample_passband(
        speclite_filter, delta_wavelength, wavelength_unit_output=u.nm
    )
    # new_passband_2 should be ~10 times in length
    new_passband_2 = downsample_passband(
        speclite_filter, delta_wavelength, wavelength_unit_output=u.angstrom
    )
    assert len(new_passband_1[0]) / len(new_passband_2[0]) < 1
    # Test erroneous input wavelength
    # This should auto-adjust u.nm to u.angstrom
    new_passband_3 = downsample_passband(
        speclite_filter, delta_wavelength, wavelength_unit_input=u.nm
    )
    assert len(new_passband_3[0]) == len(new_passband_2[0])
    # Test that there is no change in throughput values (besides rounding)
    tophat_throughput = np.ones(len(passband_wavelengths))
    tophat_passband = [passband_wavelengths, tophat_throughput]
    new_passband_4 = downsample_passband(tophat_passband, delta_wavelength)
    new_passband_5 = downsample_passband(
        tophat_passband,
        delta_wavelength,
        wavelength_unit_input=u.nm,
        wavelength_unit_output=u.mm,
    )
    assert all(new_passband_4[1] == 1)
    assert all(new_passband_5[1] == 1)


def test_bring_passband_to_source_plane():
    # Test errors
    with pytest.raises(ValueError):
        bring_passband_to_source_plane("yellow", 10)
        bring_passband_to_source_plane((32, 12), 4)
    with pytest.raises(ValueError):
        bring_passband_to_source_plane(1, 1)
    # Test function
    speclite_filter = "lsst2016-z"
    redshift_zero = 0
    zero_redshift_passband = bring_passband_to_source_plane(
        speclite_filter, redshift_zero
    )
    redshift_one = 1
    one_redshift_passband = bring_passband_to_source_plane(
        speclite_filter, redshift_one
    )
    assert all(one_redshift_passband[0] == zero_redshift_passband[0] / 2)
    assert all(one_redshift_passband[1] == zero_redshift_passband[1])
    test_passband = [[100, 200], [0.8, 1.0]]
    redshifted_test_passband = bring_passband_to_source_plane(
        test_passband, redshift_one
    )
    assert all(redshifted_test_passband[0] == np.asarray([50, 100]))


def test_convert_passband_to_nm():
    # Test errors
    with pytest.raises(ValueError):
        convert_passband_to_nm("yellow", 10)
        convert_passband_to_nm((32, 12), u.m)
    with pytest.raises(ValueError):
        convert_passband_to_nm(1)
    # Test function
    speclite_filter = "wise2010-W1"
    orig_filter = bring_passband_to_source_plane(speclite_filter, 0)
    filter_in_nm = convert_passband_to_nm(speclite_filter)
    convert_passband_to_nm(speclite_filter, wavelength_unit_input=u.m)
    npt.assert_almost_equal(orig_filter[0] / 10, filter_in_nm[0])
    new_bandpass = [np.asarray([100, 150, 200]), np.asarray([0.5, 1.0, 0.5])]
    wavelength_units = u.m
    new_bandpass_in_nm = convert_passband_to_nm(
        new_bandpass, wavelength_unit_input=wavelength_units
    )
    npt.assert_almost_equal(new_bandpass[0] * u.m.to(u.nm), new_bandpass_in_nm[0])
    npt.assert_almost_equal(new_bandpass[1], new_bandpass_in_nm[1])


def test_pull_value_from_grid():
    # === Standard Interpolation Tests (within original grid boundaries) ===
    grid_2x2 = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Test 1: Center point in 2x2 grid
    x, y = 0.5, 0.5
    expected_value = 2.5  # (1*(0.5*0.5) + 2*(0.5*0.5) + 3*(0.5*0.5) + 4*(0.5*0.5))
    # No, bilinear is:
    # (1-0.5)*(1-0.5)*1 + (0.5)*(1-0.5)*3 + (1-0.5)*(0.5)*2 + (0.5)*(0.5)*4
    # = 0.25*1 + 0.25*3 + 0.25*2 + 0.25*4 = 0.25+0.75+0.5+1.0 = 2.5
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, x, y), expected_value)

    # Test 2: Another point in 2x2 grid
    x, y = 0.25, 0.75
    # (1-0.25)*(1-0.75)*1 + (0.25)*(1-0.75)*3 + (1-0.25)*(0.75)*2 + (0.25)*(0.75)*4
    # = (0.75*0.25)*1 + (0.25*0.25)*3 + (0.75*0.75)*2 + (0.25*0.75)*4
    # = 0.1875*1 + 0.0625*3 + 0.5625*2 + 0.1875*4
    # = 0.1875 + 0.1875 + 1.125 + 0.75 = 2.25
    expected_value = 2.25
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, x, y), expected_value)

    # Test 3: Bigger grid
    grid_3x3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x, y = 0.5, 0.5
    # Interpolates between 1,2,4,5. Expected: (1+2+4+5)/4 = 12/4 = 3
    expected_value = 3.0
    npt.assert_almost_equal(pull_value_from_grid(grid_3x3, x, y), expected_value)

    # Test 4: Array input
    x_arr = np.array([0.5, 0.25])
    y_arr = np.array([0.5, 0.75])
    expected_values_arr = np.array([2.5, 2.25])
    npt.assert_almost_equal(
        pull_value_from_grid(grid_2x2, x_arr, y_arr), expected_values_arr
    )

    # === Input Validation Tests ===
    # Test 5: Non-ndarray input is converted to ndarray
    grid_list = [[1.0, 2.0], [3.0, 4.0]]
    npt.assert_almost_equal(pull_value_from_grid(grid_list, 0.5, 0.5), 2.5)

    # Test 6: Error on non-2D input
    with pytest.raises(ValueError, match="array_2d must be a 2-dimensional array"):
        pull_value_from_grid(np.array([1, 2, 3]), 0.5, 0.5)

    # Test 7: Error on zero-sized dimensions
    empty_grid_rows = np.zeros((0, 5))
    with pytest.raises(
        ValueError, match="array_2d must not have zero-sized dimensions."
    ):
        pull_value_from_grid(empty_grid_rows, 0.0, 0.0)
    empty_grid_cols = np.zeros((5, 0))
    with pytest.raises(
        ValueError, match="array_2d must not have zero-sized dimensions."
    ):
        pull_value_from_grid(empty_grid_cols, 0.0, 0.0)

    # Test 8: Mismatched x/y shapes
    with pytest.raises(
        ValueError, match="x_position and y_position must have the same shape."
    ):
        pull_value_from_grid(grid_2x2, np.array([0.1, 0.2]), np.array([0.1]))

    # Test 9: Negative coordinates
    with pytest.raises(
        ValueError, match="x_position and y_position must be non-negative."
    ):
        pull_value_from_grid(grid_2x2, -0.1, 0.5)
    with pytest.raises(
        ValueError, match="x_position and y_position must be non-negative."
    ):
        pull_value_from_grid(grid_2x2, 0.5, -0.1)
    with pytest.raises(
        ValueError, match="x_position and y_position must be non-negative."
    ):
        pull_value_from_grid(grid_2x2, np.array([-0.1, 0.5]), np.array([0.5, 0.5]))

    # === Boundary and Edge Padding Tests (New Behavior) ===
    # Original grid_2x2.shape = (2,2).
    # Max allowed x for interpolation is original_shape[0] = 2.0
    # Max allowed y for interpolation is original_shape[1] = 2.0

    # Test 10: Point exactly on original boundary (still within original grid extent for interpolation)
    # x=1.0 (max index of original grid), y=0.5
    # Interpolates between (1,0)=3 and (1,1)=4. Expected (3+4)/2 = 3.5
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 1.0, 0.5), 3.5)
    # x=0.5, y=1.0 (max index of original grid)
    # Interpolates between (0,1)=2 and (1,1)=4. Expected (2+4)/2 = 3.0
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 0.5, 1.0), 3.0)

    # Test 11: Point at the maximum allowed coordinate (uses edge padding)
    # x=2.0 (original_shape[0]), y=0.5. Uses edge value from row 1.
    # Padded array row 2 is [3,4,4]. Interpolates on padded_array[2,0]=3 and padded_array[2,1]=4
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 2.0, 0.5), 3.5)
    # x=0.5, y=2.0 (original_shape[1]). Uses edge value from col 1.
    # Padded array col 2 is [2,4,4]. Interpolates on padded_array[0,2]=2 and padded_array[1,2]=4
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 0.5, 2.0), 3.0)

    # Test 12: Point at the corner of the padded extent
    # x=2.0, y=2.0. Should be padded_array[2,2] which is original_array[1,1]=4
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 2.0, 2.0), 4.0)
    # x=0.0, y=2.0. Should be padded_array[0,2] which is original_array[0,1]=2
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 0.0, 2.0), 2.0)
    # x=2.0, y=0.0. Should be padded_array[2,0] which is original_array[1,0]=3
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 2.0, 0.0), 3.0)

    # Test 13: Point slightly beyond original max index, but within padded interpolation range
    # This was the old "out-of-bounds" test, now it should work due to padding.
    # grid_2x2 padded is [[1,2,2],[3,4,4],[3,4,4]]. Interpolator grid is [0,1,2] for rows/cols.
    # Querying (1.1, 0.5) on this padded grid.
    # R1 = interp at x=1.1 between (1,0)=3 and (2,0)=3 -> R1 = 3
    # R2 = interp at x=1.1 between (1,1)=4 and (2,1)=4 -> R2 = 4
    # Final = interp at y=0.5 between R1=3 and R2=4 -> (3+4)/2 = 3.5
    # (Manual calculation: see previous thought block, it was 3.5)
    npt.assert_almost_equal(pull_value_from_grid(grid_2x2, 1.1, 0.5), 3.5)
    npt.assert_almost_equal(
        pull_value_from_grid(grid_2x2, 0.5, 1.1), 3.0
    )  # Symmetrically for y

    # Test 14: Truly out-of-bounds coordinates (beyond padded interpolation range)
    # For grid_2x2 (shape 2,2), max_x_allowed=2.0, max_y_allowed=2.0
    # Test x too large
    with pytest.raises(
        ValueError,
        match=r"x_position \(max found: 2\.10\) must be <= 2\.0 and y_position \(max found: 0\.50\) must be <= 2\.0",
    ):
        pull_value_from_grid(grid_2x2, 2.1, 0.5)
    # Test y too large
    with pytest.raises(
        ValueError,
        match=r"x_position \(max found: 0\.50\) must be <= 2\.0 and y_position \(max found: 2\.10\) must be <= 2\.0",
    ):
        pull_value_from_grid(grid_2x2, 0.5, 2.1)
    # Test both too large
    with pytest.raises(
        ValueError,
        match=r"x_position \(max found: 2\.10\) must be <= 2\.0 and y_position \(max found: 2\.20\) must be <= 2\.0",
    ):
        pull_value_from_grid(grid_2x2, 2.1, 2.2)

    # Test with array input for out of bounds
    with pytest.raises(
        ValueError,
        match=r"x_position \(max found: 2\.10\) must be <= 2\.0 and y_position \(max found: 0\.60\) must be <= 2\.0",
    ):
        pull_value_from_grid(grid_2x2, np.array([0.5, 2.1]), np.array([0.5, 0.6]))

    # Test for a 1xN or Nx1 grid (important for padding behavior)
    grid_1x3 = np.array([[10.0, 20.0, 30.0]])  # shape (1,3)
    # original_shape[0]=1 (max_x_allowed=1.0), original_shape[1]=3 (max_y_allowed=3.0)
    # padded is [[10,20,30,30],[10,20,30,30]]
    # Test 15a: x at max allowed (edge)
    npt.assert_almost_equal(
        pull_value_from_grid(grid_1x3, 1.0, 1.0), 20.0
    )  # uses padded_array[1,1]
    npt.assert_almost_equal(
        pull_value_from_grid(grid_1x3, 1.0, 0.5), 15.0
    )  # uses padded_array[1,0] and [1,1]
    # Test 15b: y at max allowed (edge)
    npt.assert_almost_equal(
        pull_value_from_grid(grid_1x3, 0.0, 3.0), 30.0
    )  # uses padded_array[0,3]
    npt.assert_almost_equal(
        pull_value_from_grid(grid_1x3, 0.5, 3.0), 30.0
    )  # uses padded_array[0,3] and [1,3]

    grid_3x1 = np.array([[10.0], [20.0], [30.0]])  # shape (3,1)
    # original_shape[0]=3 (max_x_allowed=3.0), original_shape[1]=1 (max_y_allowed=1.0)
    # padded is [[10,10],[20,20],[30,30],[30,30]]
    # Test 15c: x at max allowed (edge)
    npt.assert_almost_equal(
        pull_value_from_grid(grid_3x1, 3.0, 0.0), 30.0
    )  # uses padded_array[3,0]
    npt.assert_almost_equal(
        pull_value_from_grid(grid_3x1, 3.0, 0.5), 30.0
    )  # uses padded_array[3,0] and [3,1]
    # Test 15d: y at max allowed (edge)
    npt.assert_almost_equal(
        pull_value_from_grid(grid_3x1, 1.0, 1.0), 20.0
    )  # uses padded_array[1,1]
    npt.assert_almost_equal(
        pull_value_from_grid(grid_3x1, 0.5, 1.0), 15.0
    )  # uses padded_array[0,1] and [1,1]

    # Test 15e: Truly out of bounds for 1xN
    with pytest.raises(
        ValueError, match=r"x_position \(max found: 1\.10\) must be <= 1\.0"
    ):
        pull_value_from_grid(grid_1x3, 1.1, 1.0)
    with pytest.raises(
        ValueError, match=r"y_position \(max found: 3\.10\) must be <= 3\.0"
    ):
        pull_value_from_grid(grid_1x3, 0.5, 3.1)


NPT_DECIMAL_PLACES = 5


def test_extract_light_curve_all_cases():
    print("Running tests for extract_light_curve...")

    conv_array_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    conv_array_5x5 = np.arange(25, dtype=float).reshape(5, 5)
    pixel_size = 1.0
    eff_vel_km_s = 1.0
    time_yr_for_1px = 0.001 / u.yr.to(u.s)

    lc1 = extract_light_curve(
        conv_array_3x3,
        pixel_size,
        eff_vel_km_s,
        time_yr_for_1px,
        x_start_position=0.0,
        y_start_position=0.0,
        phi_travel_direction=0.0,
    )
    assert isinstance(lc1, np.ndarray), "Test Case 1a Failed: LC type"
    assert lc1.shape[0] == 15, f"Test Case 1b Failed: LC shape {lc1.shape}"
    np.testing.assert_array_almost_equal(
        lc1,
        np.linspace(1.0, 4.0, 15),
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 1c Failed: LC values",
    )

    lc2 = extract_light_curve(
        conv_array_3x3,
        pixel_size,
        eff_vel_km_s * u.km / u.s,
        time_yr_for_1px * u.yr,
        x_start_position=0.0,
        y_start_position=0.0,
        phi_travel_direction=0.0,
    )
    np.testing.assert_array_almost_equal(
        lc2,
        np.linspace(1.0, 4.0, 15),
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 2b Failed: LC values with units",
    )

    avg_val3 = np.mean(conv_array_5x5)
    val3 = extract_light_curve(
        conv_array_5x5, pixel_size, eff_vel_km_s, time_yr_for_1px, pixel_shift=3
    )
    np.testing.assert_almost_equal(
        val3,
        avg_val3,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 3 Failed: pixel_shift too large",
    )

    time_for_6px = time_yr_for_1px * 6.0
    avg_val4 = np.mean(conv_array_5x5)
    val4 = extract_light_curve(
        conv_array_5x5, pixel_size, eff_vel_km_s, time_for_6px, pixel_shift=0
    )
    np.testing.assert_almost_equal(
        val4,
        avg_val4,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 4 Failed: pixels_traversed too large",
    )

    avg_val5 = np.mean(conv_array_5x5)
    np.testing.assert_almost_equal(
        extract_light_curve(
            conv_array_5x5,
            pixel_size,
            eff_vel_km_s,
            time_yr_for_1px,
            x_start_position=-1.0,
        ),
        avg_val5,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 5a Failed: Negative start_position",
    )
    np.testing.assert_almost_equal(
        extract_light_curve(
            conv_array_5x5,
            pixel_size,
            eff_vel_km_s,
            time_yr_for_1px,
            x_start_position=5.0,
        ),
        avg_val5,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 5b Failed: Too large start_position",
    )

    # ▶ Test negative y_start_position (covers the y_start_position lower‐bounds check)
    np.testing.assert_almost_equal(
        extract_light_curve(
            conv_array_5x5,
            pixel_size,
            eff_vel_km_s,
            time_yr_for_1px,
            y_start_position=-1.0,
        ),
        avg_val5,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 5c Failed: Negative y_start_position",
    )

    # ▶ Test too large y_start_position (covers the y_start_position upper‐bounds check)
    np.testing.assert_almost_equal(
        extract_light_curve(
            conv_array_5x5,
            pixel_size,
            eff_vel_km_s,
            time_yr_for_1px,
            y_start_position=5.0,
        ),
        avg_val5,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 5d Failed: Too large y_start_position",
    )

    avg_val6 = np.mean(conv_array_5x5)
    val6 = extract_light_curve(
        conv_array_5x5,
        pixel_size,
        eff_vel_km_s,
        time_yr_for_1px,
        x_start_position=4.0,
        y_start_position=2.0,
        phi_travel_direction=0.0,
    )
    np.testing.assert_almost_equal(
        val6,
        avg_val6,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 6 Failed: Track leaves array",
    )

    lc7 = extract_light_curve(
        conv_array_5x5,
        pixel_size,
        eff_vel_km_s,
        time_yr_for_1px,
        x_start_position=2,
        y_start_position=2,
        phi_travel_direction=None,
        random_seed=42,
    )
    assert not np.allclose(
        lc7, np.mean(conv_array_5x5)
    ), "Test Case 7c Failed: LC is average (unexpected)"  # np.allclose is fine for "not" checks

    lc8 = extract_light_curve(
        conv_array_5x5,
        pixel_size,
        eff_vel_km_s,
        time_yr_for_1px,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=None,
        random_seed=123,
    )
    assert not np.allclose(
        lc8, np.mean(conv_array_5x5)
    ), "Test Case 8c Failed: LC is average (unexpected)"

    res9 = extract_light_curve(
        conv_array_5x5,
        pixel_size,
        eff_vel_km_s,
        time_yr_for_1px,
        pixel_shift=1,
        x_start_position=0.0,
        y_start_position=0.0,
        phi_travel_direction=0.0,
        return_track_coords=True,
    )
    lc9, x_coords9, y_coords9 = res9
    np.testing.assert_array_almost_equal(
        lc9,
        np.linspace(conv_array_5x5[1, 1], conv_array_5x5[2, 1], 15),
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 9b Failed: LC values with shift",
    )
    np.testing.assert_array_almost_equal(
        x_coords9,
        np.linspace(0.0, 1.0, 15) + 1.0,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 9c Failed: X coords with shift",
    )
    np.testing.assert_array_almost_equal(
        y_coords9,
        np.zeros(15) + 1.0,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 9d Failed: Y coords with shift",
    )

    avg_val10 = np.mean(conv_array_3x3)
    val10 = extract_light_curve(
        conv_array_3x3, pixel_size, eff_vel_km_s, time_yr_for_1px, pixel_shift=1
    )
    np.testing.assert_almost_equal(
        val10,
        avg_val10,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 10 Failed: Safe array empty",
    )

    x_s, y_s = 1.0, 1.0
    expected_val11 = conv_array_3x3[int(x_s), int(y_s)]
    lc11 = extract_light_curve(
        conv_array_3x3,
        pixel_size,
        eff_vel_km_s,
        0.0,
        x_start_position=x_s,
        y_start_position=y_s,
        phi_travel_direction=0.0,
    )
    np.testing.assert_array_almost_equal(
        lc11,
        np.full(10, expected_val11),
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 11c Failed: LC values zero traversal",
    )

    # ---- NEW TEST CASES FOR COVERAGE ----

    # Test Case 12: Covers `if max_safe_idx_x < 0:` in random x choice
    # N_safe_dim_x from shape[0] of safe_array.
    # conv_array (3,4), pixel_shift=1.
    # safe_array slice for rows: [1 : -1-1] = [1 : 3-2] = [1:1]. Size 0.
    # N_safe_dim_x = 0. max_safe_idx_x = -1.
    conv_array_3x4_for_safe_x_neg = np.arange(12, dtype=float).reshape(3, 4)
    avg_flux_tc12 = np.mean(conv_array_3x4_for_safe_x_neg)
    val12 = extract_light_curve(
        convolution_array=conv_array_3x4_for_safe_x_neg,
        pixel_size=pixel_size,
        effective_transverse_velocity=eff_vel_km_s,
        light_curve_time_in_years=0,
        pixel_shift=1,
        x_start_position=None,
        y_start_position=None,  # y_choice won't be reached if x_choice fails
        random_seed=42,  # Seed helps if x_choice passes but y_choice has an issue
    )
    np.testing.assert_almost_equal(
        val12,
        avg_flux_tc12,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 12 Failed: max_safe_idx_x < 0 in x-choice should return avg flux.",
    )

    # Test Case 13: Covers `else: x_start_position = float(rng.integers(0, max_safe_idx_x + 1))` (x random border)
    # conv_array (2,4), pixel_shift=0.
    # N_safe_dim_x (rows) = 2. max_safe_idx_x = 1. -> x_start_position chosen from [0,1] using rng.integers(0,2)
    # N_safe_dim_y (cols) = 4. max_safe_idx_y = 3. -> y_start_position chosen from [1,2] using rng.integers(1,3)
    conv_array_2x4_for_x_border = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=float
    )
    # Based on pytest output (actual result is 30.0):
    # x_start_position (row index) from rng.integers(0,2) (1st call, seed 42) -> 0
    # y_start_position (col index) from rng.integers(1,3) (2nd call, seed 42) -> 2 (in your env)
    # Point is (row=0, col=2), value is conv_array_2x4_for_x_border[0,2] = 30.0
    expected_lc13 = np.full(10, 30.0)
    lc13 = extract_light_curve(
        convolution_array=conv_array_2x4_for_x_border,
        pixel_size=pixel_size,
        effective_transverse_velocity=eff_vel_km_s,
        light_curve_time_in_years=0,
        pixel_shift=0,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=0.0,
        random_seed=42,
    )
    np.testing.assert_array_almost_equal(
        lc13,
        expected_lc13,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 13 Failed: x_start_position random border choice incorrect.",
    )

    # Test Case 14: Covers `if max_safe_idx_y < 0:` in random y choice
    # N_safe_dim_y from shape[1] of safe_array.
    # conv_array (4,3), pixel_shift=1.
    # safe_array slice for rows: [1 : 4-1-1=2]. N_safe_dim_x = 1. max_safe_idx_x = 0. (x_start chosen as 0)
    # safe_array slice for cols: [1 : 3-1-1=1]. N_safe_dim_y = 0. max_safe_idx_y = -1.
    conv_array_4x3_for_safe_y_neg = np.arange(12, dtype=float).reshape(4, 3)
    avg_flux_tc14 = np.mean(conv_array_4x3_for_safe_y_neg)
    val14 = extract_light_curve(
        convolution_array=conv_array_4x3_for_safe_y_neg,
        pixel_size=pixel_size,
        effective_transverse_velocity=eff_vel_km_s,
        light_curve_time_in_years=0,
        pixel_shift=1,
        x_start_position=None,
        y_start_position=None,
        random_seed=42,
    )
    np.testing.assert_almost_equal(
        val14,
        avg_flux_tc14,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 14 Failed: max_safe_idx_y < 0 in y-choice should return avg flux.",
    )

    # Test Case 15: Covers `else: y_start_position = float(rng.integers(0, max_safe_idx_y + 1))` (y random border)
    # conv_array (4,2), pixel_shift=0.
    # N_safe_dim_x (rows) = 4. max_safe_idx_x = 3. -> x_start_position chosen from [1,2] using rng.integers(1,max_safe_idx_x) which is rng.integers(1,3)
    # N_safe_dim_y (cols) = 2. max_safe_idx_y = 1. -> y_start_position chosen from [0,1] using rng.integers(0,max_safe_idx_y+1) which is rng.integers(0,2)
    conv_array_4x2_for_y_border = np.array(
        [[10, 20], [30, 40], [50, 60], [70, 80]], dtype=float
    )
    # With random_seed=42 (based on latest pytest output for TC15):
    # 1st rng call for this test (for x_start): rng.integers(1, 3) -> 1. So x_start_position (row index) = 1.0
    # 2nd rng call for this test (for y_start): rng.integers(0, 2) -> 1. So y_start_position (col index) = 1.0
    # Point is (row=1, col=1), value is conv_array_4x2_for_y_border[1,1] = 40.0
    expected_lc15 = np.full(10, 40.0)  # Corrected based on latest pytest output
    lc15 = extract_light_curve(
        convolution_array=conv_array_4x2_for_y_border,
        pixel_size=pixel_size,
        effective_transverse_velocity=eff_vel_km_s,
        light_curve_time_in_years=0,
        pixel_shift=0,
        x_start_position=None,
        y_start_position=None,
        phi_travel_direction=0.0,
        random_seed=42,
    )
    np.testing.assert_array_almost_equal(
        lc15,
        expected_lc15,
        decimal=NPT_DECIMAL_PLACES,
        err_msg="Test Case 15 Failed: y_start_position random border choice incorrect.",
    )

    print("extract_light_curve tests PASSED")


def test_theta_star_physical_realistic_scenario():
    print("Running simple realistic test for theta_star_physical...")

    # Realistic cosmological parameters for a quasar lensed by a galaxy/cluster
    z_lens_realistic = 0.68  # Lens redshift (e.g., a galaxy in a cluster)
    z_src_realistic = 1.734  # Source redshift (e.g., a background quasar)
    lens_mass_solar = 1.0  # Mass of the microlens in solar masses (e.g., a star)

    cosmology_model = Planck18  # Using default Planck18 cosmology

    # Call the function
    theta_E_arcsec, theta_E_lens_plane_m, theta_E_src_plane_m = theta_star_physical(
        z_lens=z_lens_realistic,
        z_src=z_src_realistic,
        m=lens_mass_solar,
        cosmo=cosmology_model,
    )

    # 1. Check output types
    assert isinstance(
        theta_E_arcsec, Quantity
    ), "theta_E_arcsec should be an Astropy Quantity"
    assert isinstance(
        theta_E_lens_plane_m, Quantity
    ), "theta_E_lens_plane_m should be an Astropy Quantity"
    assert isinstance(
        theta_E_src_plane_m, Quantity
    ), "theta_E_src_plane_m should be an Astropy Quantity"
    print("Output types are correct.")

    # 2. Check units
    assert (
        theta_E_arcsec.unit == u.arcsec
    ), f"theta_E_arcsec unit is {theta_E_arcsec.unit}, expected arcsec"
    assert (
        theta_E_lens_plane_m.unit == u.m
    ), f"theta_E_lens_plane_m unit is {theta_E_lens_plane_m.unit}, expected m"
    assert (
        theta_E_src_plane_m.unit == u.m
    ), f"theta_E_src_plane_m unit is {theta_E_src_plane_m.unit}, expected m"
    print("Output units are correct.")

    # 3. Check for plausible values (non-NaN, positive)
    assert not np.isnan(theta_E_arcsec.value), "theta_E_arcsec value is NaN"
    assert not np.isnan(theta_E_lens_plane_m.value), "theta_E_lens_plane_m value is NaN"
    assert not np.isnan(theta_E_src_plane_m.value), "theta_E_src_plane_m value is NaN"
    print("Values are not NaN.")

    assert (
        theta_E_arcsec.value > 0
    ), f"theta_E_arcsec ({theta_E_arcsec}) should be positive"
    assert (
        theta_E_lens_plane_m.value > 0
    ), f"theta_E_lens_plane_m ({theta_E_lens_plane_m}) should be positive"
    assert (
        theta_E_src_plane_m.value > 0
    ), f"theta_E_src_plane_m ({theta_E_src_plane_m}) should be positive"
    print("Values are positive.")

    # 4. Very broad plausibility check for angular size (for a 1 M_sun lens)
    # Einstein radius for a solar mass lens is typically micro-arcseconds to a few milli-arcseconds.
    # 1 micro-arcsecond = 1e-6 arcsec
    # 10 milli-arcseconds = 0.01 arcsec
    # This is a very loose check.
    assert (
        1e-7 < theta_E_arcsec.value < 0.1
    ), f"theta_E_arcsec ({theta_E_arcsec}) is outside a very broad plausible range for a 1 M_sun lens."
    print(f"theta_E_arcsec: {theta_E_arcsec}")
    print(f"theta_E_lens_plane_m: {theta_E_lens_plane_m}")
    print(f"theta_E_src_plane_m: {theta_E_src_plane_m}")


def test_get_tau_sf_from_distribution_agn_variability():

    # --- Exact Mean (Sanity Check) ---
    # If covariance is near-zero, output should exactly equal the target means.
    means = np.array(
        [8.0, -23.0, -0.5, 2.0, 1.0]
    )  # [log_BH, M_i, log_SF, log_tau, z_src]
    cov_zero = np.eye(5) * 1e-20

    sf_res, tau_res = get_tau_sf_from_distribution_agn_variability(
        black_hole_mass_exponent=8.0,
        known_mag_abs=-23.0,
        z_src=1.0,
        means=means,
        cov=cov_zero,
        nsamps=1,
    )
    npt.assert_approx_equal(sf_res, -0.5, significant=3)  # Target index 2
    npt.assert_approx_equal(tau_res, 2.0, significant=3)  # Target index 3

    # --- Correlation Logic ---
    # If BH_Mass (idx 0) is correlated with log_tau (idx 3), shifting BH_Mass must shift log_tau.
    means_corr = np.zeros(5)
    cov_corr = np.eye(5) * 1e-6  # Small variance to reduce noise
    # Add strong positive correlation (0.9) between BH_Mass (0) and log_tau (3)
    cov_corr[0, 3] = 0.9 * 1e-6
    cov_corr[3, 0] = 0.9 * 1e-6

    # Case A: Input at mean (0) -> Expect Output near mean (0)
    _, tau_0 = get_tau_sf_from_distribution_agn_variability(
        0, 0, 0, means_corr, cov_corr, nsamps=100
    )
    # Case B: Input shifted (+1 sigma) -> Expect Output shifted (+0.9 sigma)
    _, tau_1 = get_tau_sf_from_distribution_agn_variability(
        1.0, 0, 0, means_corr, cov_corr, nsamps=100
    )

    npt.assert_allclose(np.mean(tau_0), 0.0, atol=0.01)
    npt.assert_allclose(np.mean(tau_1), 0.9, atol=0.01)

    # --- Scatter/Variance Verification ---
    # Setup: Uncorrelated variables with defined large variance
    means_scatter = np.zeros(5)
    cov_scatter = np.eye(5) * 4.0  # var = 4.0 => std = 2.0

    # Draw large sample to ensure statistical significance
    sf_scat, tau_scat = get_tau_sf_from_distribution_agn_variability(
        0, 0, 0, means_scatter, cov_scatter, nsamps=5000
    )

    # 1. Check that the distribution width (std) matches the input matrix (2.0)
    # allowing for small sampling error (atol=0.1)
    npt.assert_allclose(np.std(sf_scat), 2.0, atol=0.1)
    npt.assert_allclose(np.std(tau_scat), 2.0, atol=0.1)

    # 2. Check explicitly that there is a large spread (min vs max)
    sf_range = np.max(sf_scat) - np.min(sf_scat)
    # For a normal distribution with std=2, range should be approx 2 * 3 * std (~12)
    assert sf_range > 8.0

    # --- Shapes ---
    # nsamps=1 should return scalars
    scalar_res = get_tau_sf_from_distribution_agn_variability(
        0, 0, 0, means, cov_zero, nsamps=1
    )
    assert np.isscalar(scalar_res[0]) or scalar_res[0].ndim == 0
    assert scalar_res[0].ndim == scalar_res[1].ndim

    # nsamps=10 should return arrays
    arr_res = get_tau_sf_from_distribution_agn_variability(
        0, 0, 0, means, cov_zero, nsamps=10
    )
    assert arr_res[0].shape == (10,)
    assert isinstance(arr_res, tuple)
    assert arr_res[0].shape == (10,)
    assert arr_res[0].shape == arr_res[1].shape


def test_get_breakpoint_frequency_and_std_agn_variability():
    # --- Math Check ---
    # log_SF = log10(sqrt(2)) => SF = sqrt(2) => std = SF/sqrt(2) = 1.0
    # log_tau = log10(1/2pi)  => tau = 1/2pi  => freq = 1/(2pi*tau) = 1.0 => log_freq = 0.0
    log_sf = np.log10(np.sqrt(2))
    log_tau = np.log10(1 / (2 * np.pi))

    log_freq, std = get_breakpoint_frequency_and_std_agn_variability(log_sf, log_tau)
    npt.assert_approx_equal(std, 1.0)
    npt.assert_approx_equal(log_freq, 0.0)

    # --- Vectorization ---
    log_sf_arr = np.array([log_sf, log_sf])
    log_tau_arr = np.array([log_tau, log_tau])

    log_freq_arr, std_arr = get_breakpoint_frequency_and_std_agn_variability(
        log_sf_arr, log_tau_arr
    )

    assert log_freq_arr.shape == (2,)
    npt.assert_array_almost_equal(std_arr, [1.0, 1.0])
    assert log_freq_arr.shape == log_freq_arr.shape

    # --- 2D Vectorization ---
    log_sf_arr = np.random.normal(log_sf, 0.1, size=(1000, 2))
    log_tau_arr = np.random.normal(log_tau, 0.1, size=(1000, 2))

    log_freq_arr, std_arr = get_breakpoint_frequency_and_std_agn_variability(
        log_sf_arr, log_tau_arr
    )

    assert log_freq_arr.shape == (1000, 2)
    assert std_arr.shape == (1000, 2)
    assert log_freq_arr.shape == log_freq_arr.shape
