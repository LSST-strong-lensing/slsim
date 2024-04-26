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
    define_bending_power_law_psd,
    define_frequencies,
    normalize_light_curve,
    generate_signal,
    generate_signal_from_bending_power_law,
    generate_signal_from_generic_psd,
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
