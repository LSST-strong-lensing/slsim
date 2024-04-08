import numpy as np
from scipy.fftpack import ifft
from astropy import constants as const
from astropy import units as u
from slsim.Util.param_util import (
    amplitude_to_magnitude,
    magnitude_to_amplitude,
)


def spin_to_isco(spin):
    """Converts dimensionless spin parameter of a black hole to the innermost stable
    circular orbit in gravitational radii [R_g = GM/c^2, with units length]

    :param spin: Dimensionless spin of black hole, ranging from -1 to 1. Positive values
        represent orbits aligned with the black hole spin.
    :return: Float value of innermost stable circular orbit, ranging from 1 to 9.
    """
    if abs(spin) > 1:
        raise ValueError("Absolute value of spin cannot exceed 1")
    # Calculate intermediate values
    z1 = 1 + (1 - spin**2) ** (1 / 3) * ((1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3))
    z2 = (3 * spin**2 + z1**2) ** (1 / 2)

    # Return ISCO distance in gravitational radii
    return 3 + z2 - np.sign(spin) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2)


def calculate_eddington_luminosity(black_hole_mass_exponent):
    """Calculates the Eddington luminosity for a black hole mass exponent. The Eddington
    luminosity is the theoretical limit of the accretion rate due to radiation pressure
    for spherical (Bondi) accretion.

    Eddington_luminosity = 4 * pi * G * black_hole_mass * mass_proton
                              * c / sigma_thompson

    :param black_hole_mass_exponent: The log of the black hole mass normalized by the mass
        of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :return: Eddington luminosity
    """
    black_hole_mass = convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent)
    return 4 * np.pi * const.G * black_hole_mass * const.m_p * const.c / const.sigma_T


def eddington_ratio_to_accretion_rate(
    black_hole_mass_exponent, eddington_ratio, efficiency=0.1
):
    """Calculates the mass that must be accreted by the accretion disk for the accretion
    disk to radiate at the desired Eddington ratio.

    Bolometric_luminosity = mass_accreted * c^2 * efficiency

    :param black_hole_mass_exponent: The log of the black hole mass normalized by the
        mass of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param eddington_ratio: The desired Eddington ratio defined as a fraction of
        bolometric luminosity / Eddington luminosity.
    :param efficiency: The efficiency of mass-to-energy conversion in accretion disk
    :return: Required mass_accreted for accretion disk to radiate at the desired
        Eddington ratio
    """
    if efficiency <= 0:
        raise ValueError("Efficiency cannot be negative")

    # Calculate Eddington luminosity
    l_eddington = calculate_eddington_luminosity(black_hole_mass_exponent)

    # Calculate required accreted mass to reach Eddington luminosity
    m_eddington_accreted = l_eddington / (efficiency * const.c**2)

    return eddington_ratio * m_eddington_accreted


def calculate_gravitational_radius(black_hole_mass_exponent):
    """Calculates the gravitational radius (R_g) of a black hole. The gravitational.

    radius defines the typical size scales around a black hole for AGN.
    The formula for gravitational radius is: R_g  = G * mass / c^2

    :param black_hole_mass_exponent: The log of the black hole mass normalized by the mass
        of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :return: The gravitational radius in astropy length units.
    """
    black_hole_mass = convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent)
    return const.G * black_hole_mass / const.c**2


def convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent):
    """This function takes in the log of the black hole mass normalized by the mass of
    the sun and returns the mass of the black hole.

    :param black_hole_mass_exponent: The log of the black hole mass normalized by the
        mass of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :return: The mass of the black hole in astropy units.
    """
    return 10**black_hole_mass_exponent * const.M_sun


def thin_disk_temperature_profile(
    radial_points, black_hole_spin, black_hole_mass_exponent, eddington_ratio
):
    """Calculates the thin disk temperature profile at all given radial positions
    assuming the Shakura-Sunyaev geometricly thin, optically thick accretion disk.

    The formula for a thin disk temperature profile is:

        T(r) = (3 * G * black_hole_mass * accretion_rate * (1 - (r_min / r)**0.5)
                / (8 * pi * sigma_sb * r^3))^0.25

    :param radial_points: A list of the radial positions in gravitational units
        to calculate the temperature at.
    :param black_hole_spin: The dimensionless spin parameter of the black hole, where
        the spinless case (spin = 0) corresponds to a Schwarzschild black hole.
        Positive spin represents the accretion disk's angular momentum is aligned
        with the black hole's spin, and negative spin represents retrograde accretion
        flow.
    :param black_hole_mass_exponent: The log of the black hole mass normalized by the mass
        of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param eddington_ratio: The fraction of the eddington limit which the black hole
        is accreting at.
    :return: The temperature of the accretion disk at all radii in units Kelvin.
    """
    isco_radius = spin_to_isco(black_hole_spin)
    black_hole_mass = convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent)
    accretion_rate = eddington_ratio_to_accretion_rate(
        black_hole_mass_exponent, eddington_ratio
    )
    gravitational_radius = calculate_gravitational_radius(black_hole_mass_exponent)

    # Set up a mask so all radial values less than the ISCO have zero temperature
    mask = radial_points >= isco_radius

    # Calculate the part which does not involve any inputs
    multiplicative_constant = 3 * const.G / (8 * np.pi * const.sigma_sb)

    # Calculate the part dependent on inputs
    dynamic_values = (
        black_hole_mass
        * accretion_rate
        * (1 - (isco_radius / radial_points) ** 0.5)
        / (radial_points * gravitational_radius) ** 3
    )

    return np.nan_to_num(mask * (multiplicative_constant * dynamic_values) ** 0.25)


def planck_law(temperature, wavelength_in_nanometers):
    """This takes a temperature in Kelvin and a wavelength in nanometers, and returns
    the spectral radiance of the object as if it emitted black body radiation. This is
    the spectral radiance per wavelength as opposed to per frequency, leading to
    dependence as wavelength^(-5).

    Planck's law states:

    B(T, lam) = (2 * h * c^2 / lam^5) * (1 / (e^(h * c / (lam * k_B * T)) - 1))

    :param temperature: Temperature of the black body, in [Kelvin]
    :param wavelength_in_nanometers: Emitted wavelength in local rest frame in [nanometers]
    :return: The spectral radiance of the black body
    """
    # If wavelength was entered as an int or float, append units
    if type(wavelength_in_nanometers) != u.Quantity:
        wavelength_in_nanometers *= u.nm
    if type(temperature) != u.Quantity:
        temperature *= u.K

    e_exponent = (
        const.h * const.c / (wavelength_in_nanometers * const.k_B * temperature)
    )
    prefactor = 2 * const.h * const.c**2 / wavelength_in_nanometers**5.0

    return prefactor / (np.e ** (e_exponent) - 1)


def planck_law_derivative(temperature, wavelength_in_nanometers):
    """This numerically approximates the derivative of the spectral radiance with
    respect to temperature in Kelvin.

    Numerically calculating this derivative uses the limit definition of a derivative.

    d(f(x)) ~ lim (delta_x -> 0) (f(x+delta_x) - f(x)) / delta_x

    :param temperature: Temperature of the black body, in [Kelvin].
    :param wavelength_in_nanometers: Emitted wavelength in local rest frame in [nanometers].
    :return: The derivative of the spectral radiance with respect to temperature for a black body.
    """
    if type(temperature) == u.Quantity:
        temperature = temperature.value

    return (
        planck_law(temperature + 1e-2, wavelength_in_nanometers)
        - planck_law(temperature, wavelength_in_nanometers)
    ) / 1e-2


def create_radial_map(r_out, r_resolution, inclination_angle):
    """This creates a 2-dimentional array of radial positions where the maximum radius
    is defined by r_out, and the radial resolution is defined by r_resolution.

    :param r_out: The maximum radial value in [R_g]. For an accretion disk, this can be 10^3
        to 10^5.
    :param r_resolution: The number of points between r = 0 and r = r_out. The final map
        will be shape (2 * r_resolution), (2 * r_resolution).
    :param inclination_angle: The inclination of the plane of the accretion disk with
        respect to the observer in [degrees].
    :return: A 2D array of radial positions of shape ((2 * r_resolution), (2
        * r_resolution)) in the projected plane of the sky.
    """
    x_values = np.linspace(-r_out, r_out, 2 * r_resolution)
    y_values = np.linspace(-r_out, r_out, 2 * r_resolution) / np.cos(
        np.radians(inclination_angle)
    )

    X, Y = np.meshgrid(x_values, y_values)

    return (X**2.0 + Y**2.0) ** 0.5


def create_phi_map(r_out, r_resolution, inclination_angle):
    """This creates a 2-dimentional array of phi values at all positions where the
    maximum radius is defined by r_out, and the radial resolution is defined by
    r_resolution.

    :param r_out: The maximum radial value in [R_g]. For an accretion disk, this can be
        10^3 to 10^5.
    :param r_resolution: The number of points between r = 0 and r = r_out. The final map
        will be shape (2 * r_resolution), (2 * r_resolution)
    :param inclination_angle: The inclination of the plane of the accretion disk with
        respect to the observer in [degrees].
    :return: A 2-dimensional array of phi values at radial positions of shape ((2 *
        r_resolution), (2 * r_resolution)) in the projected plane of the sky, such that
        phi = 0 is nearest to the observer.
    """
    x_values = np.linspace(-r_out, r_out, 2 * r_resolution)
    y_values = np.linspace(-r_out, r_out, 2 * r_resolution) / np.cos(
        np.radians(inclination_angle)
    )

    X, Y = np.meshgrid(x_values, y_values)
    # must add pi/2 so phi = 0 points towards observer.
    return (np.arctan2(Y, X) + np.pi / 2) % (2 * np.pi)


def calculate_time_delays_on_disk(
    radial_map, phi_map, inclination_angle, corona_height
):
    """This calculates the time lags due to light travel time from a point source
    located above the black hole to simulate the lamppost geometry. The corona is
    defined as the point-souce approximation of the X-ray variable source.

    The light travel time lags, tau(r, phi), are defined in the lamppost geometry through:

    c * tau(r, phi|h_corona, inclination) = sqrt(h_corona^2 + r^2) +
                                    h_corona * cos(inclination) - r * sin(inclination) * cos(phi)

    :param radial_map: A 2-dimension array of radial values on the accretion disk in units [R_g].
    :param inclination_angle: The tilt of the accretion disk with respect to the observer in [degrees].
        Zero degrees is face on, 90 degrees is edge on.
    :param corona_height: The height of the corona in gravitational_radii. Typical choices range
        from 0 to 100 [R_g].
    :return: A 2-dimensional array of time delays between the corona and the accretion disk in
        units of [R_g / c].
    """
    return (
        (radial_map**2.0 + corona_height**2.0) ** 0.5
        + corona_height * np.cos(np.radians(inclination_angle))
        - radial_map * np.cos(phi_map) * np.sin(np.radians(inclination_angle))
    )


def calculate_geometric_contribution_to_lamppost_model(radial_map, corona_height):
    """This calculates the geometric scaling factor of the X-ray luminosity (L_x) as
    seen by the accretion disk due to the varying distances involved in the lamppost
    model.

    According to Equation (2) of Cackett+ (2007), the scaling factor of L_x due to the
    geometry of the system is:

    (1-A) cos(theta_x) / (4 * pi * sigma_sb * R_{*}^{2})

    Where A is the albedo of the disk (taken to be zero for complete absorption),
    theta_x is the angle of incidence between the X-rays and the disk (zero is normal to the disk),
    cos(theta_x) is taken to be h_corona / R_{*} for a geometrically flat disk,
    sigma_sb is the Stefan-Boltzmann constant,
    and R_{*} is the distance between the X-ray source and any part of the disk.

    :param radial_map: A 2-dimension array of radial values on the accretion disk in units [R_g],
        potentially generated with the create_radial_map function above.
    :param corona_height: The height of the corona in units [R_g]. Typical choices range
        from 0 to 100.
    :return: A 2-dimensional array of multiplicative factors representing the impact of
        geometry between the lamppost source and the accretion disk acting on the fluctuations
        of the X-ray source.
    """
    # Add 0.5 R_g to the point source to avoid dividing by zero
    corona_height += 0.5
    distance_to_xray_source = (radial_map**2.0 + corona_height**2.0) ** 0.5
    cosine_theta_x = corona_height / distance_to_xray_source

    return cosine_theta_x / (
        4.0 * np.pi * const.sigma_sb.value * distance_to_xray_source**2.0
    )


def calculate_dt_dlx(radial_map, temperature_map, corona_height):
    """This approximates the change in temperature on the accretion disk due to a change
    in the incident X-ray flux from the source in the lamppost model.

    This approximation follows that the change of temperature due to a change in X-ray
    energy is small:

    (1) temp_disk^4 = temp_viscous^4 + mean_energy_irradiation * geometric_factor

    (2) (temp_disk + delta_temp)^4
        = temp_viscous^4 + (mean_energy_irradiation + delta_Lx) * geometric_factor

    (3) temp_disk^4 + 4 * delta_temp * temp_disk^3 + (order delta_temp^2...)
        = temp_disk^4 + delta_Lx * geometric_factor

    (4) delta_temp / delta_Lx = geometric_factor / (4 * temp_disk^3) + higher order corrections

    (5) dt/dlx ~ geometric_factor / (4 * temp_disk^3)

    Where temp_viscous is the temperature due to the viscous nature of the black body like disk,
    mean_energy_irradiation is the average energy of the Xray source (which drops out),
    and delta_temp is the change in temperature.

    A small increase in the illuminating flux is introduced in (2).
    Terms are expanded and temp_disk^4 is reconstructed on right hand side of equation in (3).
    Terms are collected in (4), higher order terms are discarded.
    The fractional change in temperature with respect to flux is approximated
        to be the derivative in (5) as both delta_temp and delta_Lx are assumed to be small.

    :param radial_map: A 2-dimension array of radial values on the accretion disk in units [R_g].
    :param temperature_map: A 2-dimensional array of temperature values on the accretion disk
        in [Kelvin].
    :param corona_height: The height of the corona in units [R_. Typical values range
        from 0 to 100.
    :return: A 2-dimensional map of values representing the change in temperature with
        respect to a change in X-ray flux.
    """

    geometric_map = calculate_geometric_contribution_to_lamppost_model(
        radial_map, corona_height
    )

    return geometric_map / (4.0 * temperature_map**3.0)


def calculate_mean_time_lag(response_function):
    """This helper function takes in a response function (or any array) and calculates the
    weighted average following:

    weighted_average = sum(time_axis * response_function) / sum(response_function)

    The function np.nansum() is used in favor of np.sum() to avoid issues with np.nan values.

    :param response_function: The input response function (or any array) to calculate the
        weighted average from (e.g. the output of
        astro_util.calculate_accretion_disk_response_function() which represents the
        accretion disk's response function). Weighting is in the same units of spacing
        between values. For response functions generated with this code, these spacings
        are of units [R_g / c] by default.
    :return: A single value representing the weighted average. Units are equivalent to
        the x-axis spacings of the response_function (or array). For default response
        functions generated with this code, the output has units [R_g / c].
    """
    return np.nansum(
        np.linspace(0, len(response_function) - 1, len(response_function))
        * response_function
    ) / np.nansum(response_function)


def calculate_accretion_disk_response_function(
    r_out,
    r_resolution,
    inclination_angle,
    rest_frame_wavelength_in_nanometers,
    black_hole_mass_exponent,
    black_hole_spin,
    corona_height,
    eddington_ratio,
):
    """This calculates the response of the accretion disk due to a flash in the
    illuminating X-ray source in the lamppost geometry. This response function
    represents the kernel between any driving signal from the point-like source and the
    accretion disk at a specified wavelength.

    This response function is calculated by summing over all individual pixel responses
    and binning them according to their time lag with a weighting of their wavelength
    dependent response.

    Using this response function assumes:
    1) The X-ray point source drives the optical variability.
    2) The accretion disk behaves like a black body.
    3) There is no significant absorption of the X-rays between the corona and disk.
    4) The corona is compact enough that it may be treated like a point source.
    5) The accretion disk is itself flat (otherwise cosine(theta_x) != h_corona/R_{*}).
    6) The response function can be normalized.

    To use the response function:
    1) Resample the response function at the time resolution of the signal or resample the
        signal at units [R_g / c].
    2) The time axis of the response function must then be inverted as these are time lags.
    3) Take the convolution between the driving signal and the response function.
    4) The time axis of the convolution should then be shifted forward in time by the length
        of the response function to remain consistent with respect to the driving signal.

    :param r_out: The maximum radial value of the accretion disk. This typically can be chosen
        as 10^3 to 10^5 [R_g].
    :param r_resolution: The number of points between r = 0 and r = r_out. The final map will
        be shape (2 * r_resolution), (2 * r_resolution). Higher resolution leads to longer
        calculations but smoother response functions.
    :param inclination_angle: The tilt of the accretion disk with respect to the observer
        in [degrees]. Zero degrees is face on, 90 degrees is edge on.
    :param rest_frame_wavelength_in_nanometers: Wavelength in local rest frame
        in [nanometers].
    :param black_hole_mass_exponent: The log of the black hole mass normalized by the mass
        of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param black_hole_spin: The dimensionless spin parameter of the black hole, where
        the spinless case (spin = 0) corresponds to a Schwarzschild black hole.
        Positive spin represents the accretion disk's angular momentum is aligned
        with the black hole's spin, and negative spin represents retrograde accretion
        flow.
    :param corona_height: The height of the corona in gravitational_radii. Typical choices range
        from 0 to 100 [R_g].
    :param eddington_ratio: The desired Eddington ratio defined as a fraction of bolometric
        luminosity / Eddington luminosity.
    :return: The normalized response of the accretion disk as a function of time lag in
        units [R_g / c].
    """
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

    response_function = np.histogram(
        time_delay_map,
        range=(0, np.max(time_delay_map) + 1),
        bins=int(np.max(time_delay_map) + 1),
        weights=weighting_factors,
        density=True,
    )[0]

    return response_function / np.nansum(response_function)


def define_bending_power_law_psd(
    log_breakpoint_frequency, low_frequency_slope, high_frequency_slope, frequencies
):
    """This function defines the power spectrum density (PSD) of a bending power law.
    Note that bending power law is also sometimes referred to as a broken power law.

    :param log_breakpoint_frequency: The log_{10} of the breakpoint frequency where
        the power law changes slope, in units [1/days]. Typical values range between
        -3.5 and 1.0.
    :param low_frequency_slope: The (negative) log-log slope of the PSD for low frequencies when
        the power is plotted against frequency in units [1/days]. Typically ~1.0, but
        can range from 0.0 to 2.0.
    :param high_frequency_slope: the (negative) log-log slope of the PSD for high frequencies when
        the power is plotted against frequency in units [1/days]. Typically ranges from
        2.0 to 4.0, and should be a higher power than low_frequency_slope (e.g. it should
        drop off with frequency rapidly).
    :param frequencies: A numpy array or list of frequencies to calculate the PSD at.
        This array is well defined through the define_frequencies() function.
        Note that define_frequencies() will prepare minimum and maximum frequencies, and
        there will not be a "bend" in the PSD if the breakpoint frequency does not
        fall within this range.
    :return: The PSD of the bending power law defined through the input parameters.
    """

    breakpoint_frequency = 10**log_breakpoint_frequency
    bending_power_law_psd = (frequencies**-low_frequency_slope) * (
        1
        + (frequencies / breakpoint_frequency)
        ** (high_frequency_slope - low_frequency_slope)
    ) ** -1
    return bending_power_law_psd


def define_frequencies(length_of_light_curve, time_resolution):
    """This function defines the useful frequencies for generating a power spectrum
    density (PSD). Frequencies below the low frequency limit will not contribute to the
    light curve. Frequencies above the high frequency limit (the Nyquist frequency) will
    not be able to be probed with the time_resolution, and will suffer from aliasing.

    :param length_of_light_curve: The total length of the light curve to simulate, in
        units of [days]. The generated frequencies will have a 10 times lower limit than
        required, as the function generate_signal_from_psd will generate extended light
        curves to deal with periodicity issues.
    :param time_resolution: The time resolution to generate the light curve at, in units
        of [days]. This parameter defines the high frequency limit. If generating light
        curves takes too long, consider increasing this parameter to generate fewer
        frequencies.
    :return: A numpy array of the frequencies that are probed by the light curve in
        [1/days].
    """

    length_of_generated_light_curve = 10 * length_of_light_curve
    frequencies = np.linspace(
        1 / length_of_generated_light_curve,
        1 / (2 * time_resolution),
        int(length_of_generated_light_curve) + 1,
    )
    return frequencies


def normalize_light_curve(light_curve, mean_magnitude, standard_deviation=None):
    """This function takes in a light curve and redefines its mean and standard
    deviation. It may also be used to re-normalize any time series.

    :param light_curve: A time series list or array which represents a one-dimensional
        light curve. This function does not require any specific units or spacings.
    :param mean_magnitude: The new mean value of the light curve. This is done through a
        simple shifting of the y-axis.
    :param standard_deviation: The new standard deviation of the light curve. Note this
        only makes sense for a variable signal (e.g. a constant signal cannot be given a
        new standard_deviation). A negative standard deviation will invert the x and y
        axis.
    :return: A rescaled version of the original light curve, with new mean and standard
        deviation.
    """
    light_curve = np.asarray(light_curve)
    light_curve -= light_curve.mean()
    if light_curve.std() > 0 and standard_deviation is not None:
        light_curve /= light_curve.std()
    if standard_deviation != 0 and standard_deviation is not None:
        light_curve *= standard_deviation
    light_curve += mean_magnitude
    return light_curve


def generate_signal(
    length_of_light_curve,
    time_resolution,
    log_breakpoint_frequency=-2,
    low_frequency_slope=1,
    high_frequency_slope=3,
    mean_magnitude=0,
    standard_deviation=0.1,
    normal_magnitude_variance=True,
    zero_point_mag=0,
    input_freq=None,
    input_psd=None,
    seed=None,
):
    """This function creates a stochastic signal to model AGN X-ray variability. This
    may be used to generate either a bending power law signal, or a signal following any
    input power spectrum density (psd).

    :param length_of_light_curve: The total length of the light curve to simulate, in units
        of [days]. The generated signal will be 10 times longer than this to
        deal with periodicity issues which may arise.
    :param time_resolution: The time spacing between regularly sampled points in the light curve,
        in units of [days]. This parameter defines the high frequency limit of the PSD and the
        number of points defining the light curve. If generating light curves takes too long,
        consider increasing this parameter to generate fewer frequencies.
    :param log_breakpoint_frequency: The log_{10} of the breakpoint frequency as defined in
        the bending power law, in units days^{-1}. Typical values range from -3.5 to 1.0.
    :param low_frequency_slope: The (negative) log-log slope of the PSD for low frequencies when
        the power is plotted against frequency in units [1/days]. Typically ~1.0, but
        can range from 0.0 to 2.0.
    :param high_frequency_slope: The (negative) log-log slope of the PSD for high frequencies when
        the power is plotted against frequency in units [1/days]. Typically ranges from
        2.0 to 4.0, and should be a higher power than low_frequency_slope (e.g. it should
        drop off with frequency rapidly).
    :param mean_magnitude: The mean value of the light curve to simulate. The
        PSD will produce a stochastic light curve with some mean and some standard
        deviation. This parameter will fix the mean value of the output light curve.
    :param standard_deviation: The desired standard deviation (std) of the light curve's
        variability.
    :param normal_magnitude_variance: Bool, a toggle between whether variability is calculated in
        magnitude or flux units. If True, variability will be assumed to have the given standard
        deviation in magnitude. If False, variability will assume to have the given standard
        deviation in flux. Note that if False, "negative flux" becomes likely for standard
        deviation > 0.5 mag, and will return a ValueError.
        If everything is assumed to be in flux units, simply insert your mean flux for
        "mean_magnitude" and define "normal_magnitude_variance" = True.
    :param zero_point_mag: The reference amplitude to calculate the zero point magnitude.
    :param input_freq: None or an input array of frequencies in [1/days] to use to overwrite the
        frequencies generated by astro_util.generate_frequencies(). If none, no action
        will be taken. If an array of frequencies is input, this array will override
        the frequencies used o generate the signal. This must be equal length to input_psd.
        This may be useful for testing.
    :param input_psd: None or an input array representing the PSD at input_freq. If
        None, no action will be taken. If an array is input, this must be of equal length
        to the array input_freq. Then this input_psd will override the bending power law
        generated using astro_util.define_bending_power_law_psd(). This may be useful
        for defining more complex power spectrums, or other testing.
    :param seed: None or value. If a value is provided, the random seed may be defined
        within this function.
    """
    if seed is not None:
        np.random.seed(seed)

    if input_freq is not None:
        frequencies = np.asarray(input_freq)
        assert input_psd is not None
    else:
        frequencies = define_frequencies(length_of_light_curve, time_resolution)
    if input_psd is not None:
        power_spectrum_density = np.asarray(input_psd)
        assert len(input_freq) == len(power_spectrum_density)
    else:
        power_spectrum_density = define_bending_power_law_psd(
            log_breakpoint_frequency,
            low_frequency_slope,
            high_frequency_slope,
            frequencies,
        )
    random_phases = 2.0 * np.pi * np.random.random(size=len(frequencies))
    fourier_transform = np.sqrt(power_spectrum_density) * np.exp(1j * random_phases)
    fourier_transform = np.concatenate(
        (fourier_transform, fourier_transform[-2:0:-1].conjugate())
    )
    generated_light_curve = ifft(fourier_transform)[
        : length_of_light_curve // time_resolution
    ]
    if normal_magnitude_variance is False:
        amplitude_baseline = magnitude_to_amplitude(mean_magnitude, zero_point_mag)
        amplitude_value_1 = magnitude_to_amplitude(
            mean_magnitude + standard_deviation, zero_point_mag
        )
        amplitude_value_2 = magnitude_to_amplitude(
            mean_magnitude - standard_deviation, zero_point_mag
        )

        amplitude_variations = np.min(
            (
                abs(amplitude_value_1 - amplitude_baseline),
                abs(amplitude_value_2 - amplitude_baseline),
            )
        )

        intermediate_light_curve = normalize_light_curve(
            generated_light_curve, amplitude_baseline, amplitude_variations
        )
        if any(intermediate_light_curve < 0):
            raise ValueError("Warning: Amplitude variations greater than mean flux.")

        output_light_curve = amplitude_to_magnitude(
            intermediate_light_curve, zero_point_mag
        )

    else:
        output_light_curve = normalize_light_curve(
            generated_light_curve, mean_magnitude, standard_deviation
        )
    return output_light_curve.real


def generate_signal_from_bending_power_law(
    length_of_light_curve,
    time_resolution,
    log_breakpoint_frequency=-2,
    low_frequency_slope=1,
    high_frequency_slope=3,
    mean_magnitude=0,
    standard_deviation=None,
    normal_magnitude_variance=True,
    zero_point_mag=0,
    seed=None,
):
    """Uses astro_util.generate_signal_from_psd() to create an intrinsic bending power
    law signal to use as a model for X-ray variability. Creates a light curve which can
    be sampled from using sample_intrinsic_signal().

    :param length_of_light_curve: Total length of desired light curve in [days].
    :param time_resolution: The time spacing between observations in [days].
    :param log_breakpoint_frequency: The log_{10} of the characteristic breakpoint
        frequency in the bending power law. Typically between -3.5 and 1.0.
    :param low_frequency_slope: The (negative) log-log slope of the power spectrum
        density on the low frequency side of the breakpoint frequency. Typically between
        0.0 and 2.0.
    :param high_frequency_slope: The (negative) log-log slope of the power spectrum
        density on the high frequency side of the breakpoint frequency. Typically
        between 2.0 and 4.0, and higher than the low_frequency_slope.
    :param mean_magnitude: The desired mean value of the light curve.
    :param standard_deviation: The desired standard deviation of the light curve.
    :param normal_magnitude_variance: Bool, a toggle between whether variability is calculated in
        magnitude or flux units. If True, variability will be assumed to have the given standard
        deviation in magnitude. If False, variability will assume to have the given standard
        deviation in flux. Note that if False, "negative flux" becomes likely for standard
        deviation > 0.5 mag, and will return a ValueError.
        If everything is assumed to be in flux units, simply insert your mean flux for
        "mean_magnitude" and define "normal_magnitude_variance" = True.
    :param zero_point_mag: The reference amplitude to calculate the zero point magnitude.
    :param seed: The random seed to be input for reproducability.
    :return: Two arrays, the time_array and the magnitude_array of the variability.
    """
    time_array = np.linspace(
        0, length_of_light_curve - 1, length_of_light_curve // time_resolution
    )
    magnitude_array = generate_signal(
        length_of_light_curve,
        time_resolution,
        log_breakpoint_frequency=log_breakpoint_frequency,
        low_frequency_slope=low_frequency_slope,
        high_frequency_slope=high_frequency_slope,
        mean_magnitude=mean_magnitude,
        standard_deviation=standard_deviation,
        normal_magnitude_variance=normal_magnitude_variance,
        zero_point_mag=zero_point_mag,
        seed=seed,
    )
    return time_array, magnitude_array


def generate_signal_from_generic_psd(
    length_of_light_curve,
    time_resolution,
    input_frequencies,
    input_psd,
    mean_magnitude=0,
    standard_deviation=None,
    normal_magnitude_variance=True,
    zero_point_mag=0,
    seed=None,
):
    """Uses astro_util.generate_signal_from_psd() to create an intrinsic signal from any
    input power spectrum to use as a model for X-ray variability. Creates a light curve
    which can be sampled from using sample_intrinsic_signal().

    :param length_of_light_curve: Total length of desired light curve in [days].
    :param time_resolution: The time spacing between observations in [days].
    :param input_frequencies: The input frequencies that correspond to the input power
        spectrum in [1/days]. This can be generated using
        astro_util.define_frequencies().
    :param input_psd: The input power spectrum. This must be the same size as
        input_frequencies.
    :param mean_magnitude: The desired mean value of the light curve.
    :param standard_deviation: The desired standard deviation of the light curve.
    :param normal_magnitude_variance: Bool, a toggle between whether variability is
        calculated in magnitude or flux units. If True, variability will be assumed to
        have the given standard deviation in magnitude. If False, variability will
        assume to have the given standard deviation in flux. Note that if False,
        "negative flux" becomes likely for standard deviation > 0.5 mag, and will return
        a ValueError. If everything is assumed to be in flux units, simply insert your
        mean flux for "mean_magnitude" and define "normal_magnitude_variance" = True.
    :param zero_point_mag: The reference amplitude to calculate the zero point
        magnitude.
    :param seed: The random seed to be input for reproducability.
    :return: Two arrays, the time_array in [days] and the magnitude_array of the
        variability.
    """
    time_array = np.linspace(
        0, length_of_light_curve - 1, length_of_light_curve // time_resolution
    )
    magnitude_array = generate_signal(
        length_of_light_curve,
        time_resolution,
        input_freq=input_frequencies,
        input_psd=input_psd,
        mean_magnitude=mean_magnitude,
        standard_deviation=standard_deviation,
        normal_magnitude_variance=normal_magnitude_variance,
        zero_point_mag=zero_point_mag,
        seed=seed,
    )
    return time_array, magnitude_array
