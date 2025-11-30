import numpy as np
from scipy.fftpack import ifft
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Cosmology
from slsim.Util.param_util import (
    amplitude_to_magnitude,
    magnitude_to_amplitude,
)
from astropy.units.quantity import Quantity
from speclite.filters import (
    load_filter,
    FilterResponse,
)
from scipy.interpolate import RegularGridInterpolator


def spin_to_isco(spin):
    """Converts dimensionless spin parameter of a black hole to the innermost
    stable circular orbit in gravitational radii [R_g = GM/c^2, with units
    length]

    :param spin: Dimensionless spin of black hole, ranging from -1 to 1.
        Positive values represent orbits aligned with the black hole
        spin.
    :return: Float value of innermost stable circular orbit, ranging
        from 1 to 9.
    """
    if abs(spin) > 1:
        raise ValueError("Absolute value of spin cannot exceed 1")
    # Calculate intermediate values
    z1 = 1 + (1 - spin**2) ** (1 / 3) * ((1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3))
    z2 = (3 * spin**2 + z1**2) ** (1 / 2)

    # Return ISCO distance in gravitational radii
    return 3 + z2 - np.sign(spin) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2)


def calculate_eddington_luminosity(black_hole_mass_exponent):
    """Calculates the Eddington luminosity for a black hole mass exponent. The
    Eddington luminosity is the theoretical limit of the accretion rate due to
    radiation pressure for spherical (Bondi) accretion.

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
    """Calculates the mass that must be accreted by the accretion disk for the
    accretion disk to radiate at the desired Eddington ratio.

    Bolometric_luminosity = mass_accreted * c^2 * efficiency

    :param black_hole_mass_exponent: The log of the black hole mass
        normalized by the mass of the sun; black_hole_mass_exponent =
        log_10(black_hole_mass / mass_sun). Typical AGN have an exponent
        ranging from 6 to 10.
    :param eddington_ratio: The desired Eddington ratio defined as a
        fraction of bolometric luminosity / Eddington luminosity.
    :param efficiency: The efficiency of mass-to-energy conversion in
        accretion disk
    :return: Required mass_accreted for accretion disk to radiate at the
        desired Eddington ratio
    """
    if efficiency <= 0:
        raise ValueError("Efficiency cannot be negative")

    # Calculate Eddington luminosity
    l_eddington = calculate_eddington_luminosity(black_hole_mass_exponent)

    # Calculate required accreted mass to reach Eddington luminosity
    m_eddington_accreted = l_eddington / (efficiency * const.c**2)

    return eddington_ratio * m_eddington_accreted


def calculate_gravitational_radius(black_hole_mass_exponent):
    """Calculates the gravitational radius (R_g) of a black hole. The
    gravitational.

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
    """This function takes in the log of the black hole mass normalized by the
    mass of the sun and returns the mass of the black hole.

    :param black_hole_mass_exponent: The log of the black hole mass
        normalized by the mass of the sun; black_hole_mass_exponent =
        log_10(black_hole_mass / mass_sun). Typical AGN have an exponent
        ranging from 6 to 10.
    :return: The mass of the black hole in astropy units.
    """
    return 10**black_hole_mass_exponent * const.M_sun


def thin_disk_temperature_profile(
    radial_points, black_hole_spin, black_hole_mass_exponent, eddington_ratio
):
    """Calculates the thin disk temperature profile at all given radial
    positions assuming the Shakura-Sunyaev geometricly thin, optically thick
    accretion disk.

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
    """This takes a temperature in Kelvin and a wavelength in nanometers, and
    returns the spectral radiance of the object as if it emitted black body
    radiation. This is the spectral radiance per wavelength as opposed to per
    frequency, leading to dependence as wavelength^(-5).

    Planck's law states:

    B(T, lam) = (2 * h * c^2 / lam^5) * (1 / (e^(h * c / (lam * k_B * T)) - 1))

    :param temperature: Temperature of the black body, in [Kelvin]
    :param wavelength_in_nanometers: Emitted wavelength in local rest frame in [nanometers]
    :return: The spectral radiance of the black body
    """
    # If wavelength was entered as an int or float, append units
    if type(wavelength_in_nanometers) is not u.Quantity:
        wavelength_in_nanometers *= u.nm
    if type(temperature) is not u.Quantity:
        temperature *= u.K

    e_exponent = (
        const.h * const.c / (wavelength_in_nanometers * const.k_B * temperature)
    )
    prefactor = 2 * const.h * const.c**2 / wavelength_in_nanometers**5.0

    return prefactor / (np.e ** (e_exponent) - 1)


def planck_law_derivative(temperature, wavelength_in_nanometers):
    """This numerically approximates the derivative of the spectral radiance
    with respect to temperature in Kelvin.

    Numerically calculating this derivative uses the limit definition of a derivative.

    d(f(x)) ~ lim (delta_x -> 0) (f(x+delta_x) - f(x)) / delta_x

    :param temperature: Temperature of the black body, in [Kelvin].
    :param wavelength_in_nanometers: Emitted wavelength in local rest frame in [nanometers].
    :return: The derivative of the spectral radiance with respect to temperature for a black body.
    """
    if type(temperature) is u.Quantity:
        temperature = temperature.value

    return (
        planck_law(temperature + 1e-2, wavelength_in_nanometers)
        - planck_law(temperature, wavelength_in_nanometers)
    ) / 1e-2


def create_radial_map(r_out, r_resolution, inclination_angle):
    """This creates a 2-dimentional array of radial positions where the maximum
    radius is defined by r_out, and the radial resolution is defined by
    r_resolution.

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
    """This creates a 2-dimentional array of phi values at all positions where
    the maximum radius is defined by r_out, and the radial resolution is
    defined by r_resolution.

    :param r_out: The maximum radial value in [R_g]. For an accretion
        disk, this can be 10^3 to 10^5.
    :param r_resolution: The number of points between r = 0 and r =
        r_out. The final map will be shape (2 * r_resolution), (2 *
        r_resolution)
    :param inclination_angle: The inclination of the plane of the
        accretion disk with respect to the observer in [degrees].
    :return: A 2-dimensional array of phi values at radial positions of
        shape ((2 * r_resolution), (2 * r_resolution)) in the projected
        plane of the sky, such that phi = 0 is nearest to the observer.
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
    """This calculates the time lags due to light travel time from a point
    source located above the black hole to simulate the lamppost geometry. The
    corona is defined as the point-souce approximation of the X-ray variable
    source.

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
    """This calculates the geometric scaling factor of the X-ray luminosity
    (L_x) as seen by the accretion disk due to the varying distances involved
    in the lamppost model.

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
    """This approximates the change in temperature on the accretion disk due to
    a change in the incident X-ray flux from the source in the lamppost model.

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
    :param corona_height: The height of the corona in units [R]. Typical values range
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


def calculate_accretion_disk_emission(
    r_out,
    r_resolution,
    inclination_angle,
    rest_frame_wavelength_in_nanometers,
    black_hole_mass_exponent,
    black_hole_spin,
    eddington_ratio,
    return_spectral_radiance_distribution=False,
):
    """This calculates the emission of the accretion disk due to black body
    radiation. This emission is calculated by summing over all individual
    pixels.

    :param r_out: The maximum radial value of the accretion disk. This
        typically can be chosen as 10^3 to 10^5 [R_g].
    :param r_resolution: The number of points between r = 0 and r =
        r_out. The final map will be shape (2 * r_resolution), (2 *
        r_resolution). Higher resolution leads to longer calculations
        but smoother response functions.
    :param inclination_angle: The tilt of the accretion disk with
        respect to the observer in [degrees]. Zero degrees is face on,
        90 degrees is edge on.
    :param rest_frame_wavelength_in_nanometers: Wavelength in local rest
        frame in [nanometers].
    :param black_hole_mass_exponent: The log of the black hole mass
        normalized by the mass of the sun; black_hole_mass_exponent =
        log_10(black_hole_mass / mass_sun). Typical AGN have an exponent
        ranging from 6 to 10.
    :param black_hole_spin: The dimensionless spin parameter of the
        black hole, where the spinless case (spin = 0) corresponds to a
        Schwarzschild black hole. Positive spin represents the accretion
        disk's angular momentum is aligned with the black hole's spin,
        and negative spin represents retrograde accretion flow.
    :param eddington_ratio: The desired Eddington ratio defined as a
        fraction of bolometric luminosity / Eddington luminosity.
    :param return_spectral_radiance_distribution: Boolean flag to reutrn
        the distribution of spectral radiance (for True), or the sum
        of the distribution (for False).
    :return: The result of the Planck function for a distribution of
        temperatures, either as an array representing the distribution
        in the source plane or the sum of this array.
    """
    radial_map = create_radial_map(r_out, r_resolution, inclination_angle)

    temperature_map = thin_disk_temperature_profile(
        radial_map, black_hole_spin, black_hole_mass_exponent, eddington_ratio
    )

    temperature_map *= radial_map < r_out

    emission_map = planck_law(temperature_map, rest_frame_wavelength_in_nanometers)

    if return_spectral_radiance_distribution:
        return emission_map

    return np.nansum(emission_map)


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
    represents the kernel between any driving signal from the point-like source
    and the accretion disk at a specified wavelength.

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
    """This function defines the power spectrum density (PSD) of a bending
    power law. Note that bending power law is also sometimes referred to as a
    broken power law.

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
    """This function defines the useful frequencies for generating a power
    spectrum density (PSD). Frequencies below the low frequency limit will not
    contribute to the light curve. Frequencies above the high frequency limit
    (the Nyquist frequency) will not be able to be probed with the
    time_resolution, and will suffer from aliasing.

    :param length_of_light_curve: The total length of the light curve to
        simulate, in units of [days]. The generated frequencies will
        have a 10 times lower limit than required, as the function
        generate_signal_from_psd will generate extended light curves to
        deal with periodicity issues.
    :param time_resolution: The time resolution to generate the light
        curve at, in units of [days]. This parameter defines the high
        frequency limit. If generating light curves takes too long,
        consider increasing this parameter to generate fewer
        frequencies.
    :return: A numpy array of the frequencies that are probed by the
        light curve in [1/days].
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

    :param light_curve: A time series list or array which represents a
        one-dimensional light curve. This function does not require any
        specific units or spacings.
    :param mean_magnitude: The new mean value of the light curve. This
        is done through a simple shifting of the y-axis.
    :param standard_deviation: The new standard deviation of the light
        curve. Note this only makes sense for a variable signal (e.g. a
        constant signal cannot be given a new standard_deviation). A
        negative standard deviation will invert the x and y axis.
    :return: A rescaled version of the original light curve, with new
        mean and standard deviation.
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
    """This function creates a stochastic signal to model AGN X-ray
    variability. This may be used to generate either a bending power law
    signal, or a signal following any input power spectrum density (psd).

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
        : int(length_of_light_curve / time_resolution)
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
    """Uses astro_util.generate_signal_from_psd() to create an intrinsic
    bending power law signal to use as a model for X-ray variability. Creates a
    light curve which can be sampled from using sample_intrinsic_signal().

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
        0, length_of_light_curve - 1, int(length_of_light_curve / time_resolution)
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
    """Uses astro_util.generate_signal_from_psd() to create an intrinsic signal
    from any input power spectrum to use as a model for X-ray variability.
    Creates a light curve which can be sampled from using
    sample_intrinsic_signal().

    :param length_of_light_curve: Total length of desired light curve in
        [days].
    :param time_resolution: The time spacing between observations in
        [days].
    :param input_frequencies: The input frequencies that correspond to
        the input power spectrum in [1/days]. This can be generated
        using astro_util.define_frequencies().
    :param input_psd: The input power spectrum. This must be the same
        size as input_frequencies.
    :param mean_magnitude: The desired mean value of the light curve.
    :param standard_deviation: The desired standard deviation of the
        light curve.
    :param normal_magnitude_variance: Bool, a toggle between whether
        variability is calculated in magnitude or flux units. If True,
        variability will be assumed to have the given standard deviation
        in magnitude. If False, variability will assume to have the
        given standard deviation in flux. Note that if False, "negative
        flux" becomes likely for standard deviation > 0.5 mag, and will
        return a ValueError. If everything is assumed to be in flux
        units, simply insert your mean flux for "mean_magnitude" and
        define "normal_magnitude_variance" = True.
    :param zero_point_mag: The reference amplitude to calculate the zero
        point magnitude.
    :param seed: The random seed to be input for reproducability.
    :return: Two arrays, the time_array in [days] and the
        magnitude_array of the variability.
    """
    time_array = np.linspace(
        0, length_of_light_curve - 1, int(length_of_light_curve / time_resolution)
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


def get_value_if_quantity(variable):
    """Extracts the numerical value from an astropy Quantity object or returns
    the input if not a Quantity.

    This function checks if the input variable is an instance of an
    astropy Quantity. If it is, the function extracts and returns the
    numerical value of the Quantity. If the input is not a Quantity, it
    returns the input variable unchanged.

    :param variable: The variable to be checked and possibly converted.
        Can be an astropy Quantity or any other data type.
    :type variable: Quantity or any
    :return: The numerical value of the Quantity if the input is a
        Quantity; otherwise, the input variable itself.
    :rtype: float or any
    """
    if isinstance(variable, Quantity):
        return variable.value
    else:
        return variable


def cone_radius_angle_to_physical_area(radius_rad, z, cosmo):
    """Convert cone radius angle to physical area at a specified redshift.

    This function computes the physical area, in square megaparsecs
    (Mpc^2), corresponding to a specified cone radius angle at a given
    redshift. The calculation is based on the angular diameter distance,
    which is dependent on the adopted cosmological model. This is
    particularly useful in cosmological simulations and observations
    where the physical scale of structures is inferred from angular
    measurements.

    :param radius_rad: The half cone angle in radians.
    :param z: The redshift at which the physical area is calculated.
    :param cosmo: The astropy cosmology instance used for the
        conversion.
    :type radius_rad: float
    :type z: float
    :type cosmo: astropy.cosmology instance
    :return: The physical area in square megaparsecs (Mpc^2) for the
        given cone radius and redshift.
    :rtype: float :note: The calculation incorporates the angular
        diameter distance, highlighting the interplay between angular
        measurements and physical scales in an expanding universe.
    """

    physical_radius = cosmo.angular_diameter_distance(z) * radius_rad  # Mpc
    area_physical = np.pi * physical_radius**2
    return area_physical  # in Mpc2


def downsample_passband(
    passband,
    output_delta_wavelength,
    wavelength_unit_input=u.angstrom,
    wavelength_unit_output=u.angstrom,
):
    """Takes in a throughput at one resolution and outputs a throughput at a
    downsampled resolution. This will speed up calculations which must be done
    for multiple wavelengths, especially for objects which do not have
    significant changes in signal over short wavelength steps.

    :param passband: Str or List representing passband data. Either from speclite or a user
        defined passband represented as a list of lists or arrays. The first must be wavelengths,
        and the second must be the throughput of signature: [wavelength, throughput].
    :param output_delta_wavelength: Int, this represents the desired spacing between wavelengths
        of the output passband.
    :param wavelength_unit_input: Astropy unit representing the input wavelength units.
        Speclite filters default to units of angstroms.
    :param wavelength_unit_output: Astropy unit representing the output wavelength units.
    :return: List of numpy arrays representing the downsampled passband with signature
        [wavelength, throughput].
    """
    if isinstance(passband, str):
        passband = load_filter(passband)
    if isinstance(passband, FilterResponse):
        if wavelength_unit_input != passband.effective_wavelength.unit:
            print("Changing input unit to match speclite filter unit")
            wavelength_unit_input = passband.effective_wavelength.unit
        passband = [passband.wavelength, passband.response]
    if not isinstance(passband, list):
        raise ValueError(
            "Throughput must be a speclite filter or a list of throughputs."
        )
    assert output_delta_wavelength < len(passband[0])
    # Convert to desired output wavelength units
    wavelength_ratio = wavelength_unit_input.to(wavelength_unit_output)
    passband[0] = np.asarray(passband[0][:]) * wavelength_ratio
    # Determine bins
    min_wavelength = np.min(passband[0][:])
    max_wavelength = np.max(passband[0][:])
    filter_total_wavelength_coverage = max_wavelength - min_wavelength
    nbins = max(
        int(round(1 + filter_total_wavelength_coverage / output_delta_wavelength, 0)), 2
    )
    bin_edges = np.linspace(min_wavelength, max_wavelength, nbins)
    # Make throughput
    objects_per_bin = np.histogram(passband[0], bins=bin_edges)[0]
    output_wavelengths = (bin_edges[:-1] + bin_edges[1:]) / 2
    weighted_throughput = np.histogram(
        passband[0], bins=bin_edges, weights=passband[1]
    )[0]
    normalized_throughput = weighted_throughput / objects_per_bin
    return [output_wavelengths, normalized_throughput]


def bring_passband_to_source_plane(
    passband,
    redshift,
):
    """Takes in a passband and converts the wavelengths from the observer to
    source plane.

    :param passband: Str or List representing passband data. Either from speclite or a user
        defined passband represented as a list of lists or arrays. The first must be wavelengths,
        and the second must be the throughput of signature: [wavelength, throughput].
    :param redshift: Redshift of the source.
    :return: List of numpy arrays representing the passband in the source plane.
    """
    if isinstance(passband, str):
        passband = load_filter(passband)

    if isinstance(passband, FilterResponse):
        passband = [passband.wavelength, passband.response]
    if not isinstance(passband, list):
        raise ValueError(
            "Throughput must be a speclite filter or a list of throughputs."
        )
    assert redshift >= 0
    new_wavelengths = np.asarray(passband[0]) / (1 + redshift)
    return [new_wavelengths, np.asarray(passband[1])]


def convert_passband_to_nm(
    passband,
    wavelength_unit_input=u.angstrom,
):
    """Takes in a passband and converts the wavelengths to nanometers.

    :param passband: Str or List representing passband data. Either from speclite or a user
        defined passband represented as a list of lists or arrays. The first must be wavelengths,
        and the second must be the throughput of signature: [wavelength, throughput].
    :param wavelength_unit_input: Astropy unit representing the input wavelength units.
        Speclite passbands are typically in angstroms, but a user defined passband may have
        any unit convenient for their purposes.
    :return: List of numpy arrays representing the passband with units of nanometers.
    """
    if isinstance(passband, str):
        passband = load_filter(passband)
    if isinstance(passband, FilterResponse):
        if wavelength_unit_input != passband.effective_wavelength.unit:
            print("Changing input unit to match speclite filter unit")
            wavelength_unit_input = passband.effective_wavelength.unit
        passband = [passband.wavelength, passband.response]
    if not isinstance(passband, list):
        raise ValueError(
            "Throughput must be a speclite filter or a list of throughputs."
        )
    wavelength_ratio = wavelength_unit_input.to(u.nm)
    output_passband = passband.copy()
    output_passband[0] = np.asarray(passband[0][:]) * wavelength_ratio
    return output_passband


def pull_value_from_grid(array_2d, x_position, y_position):
    """This approximates the point (x_position, y_position) in a 2d array of
    values. x_position and y_position may be decimals, and are assumed to be
    measured in pixels relative to the original grid. This uses bilinear
    interpolation (powered by scipy.interpolate.RegularGridInterpolator) with
    'edge' behavior for points at or slightly beyond the original grid
    boundaries, by interpolating on an edge-padded version of the grid.

    :param array_2d: 2 dimensional array of values.
    :param x_position: x coordinate in array_2d in pixels. Valid range
        is [0, array_2d.shape[0]].
    :param y_position: y coordinate in array_2d in pixels. Valid range
        is [0, array_2d.shape[1]].
    :return: approximation of array_2d at point (x_position, y_position)
    """
    if not isinstance(array_2d, np.ndarray):
        array_2d = np.asarray(array_2d)

    if array_2d.ndim != 2:
        raise ValueError("array_2d must be a 2-dimensional array.")

    original_shape = array_2d.shape
    if original_shape[0] == 0 or original_shape[1] == 0:
        raise ValueError("array_2d must not have zero-sized dimensions.")

    x_pos_arr = np.asarray(x_position, dtype=float)
    y_pos_arr = np.asarray(y_position, dtype=float)

    if x_pos_arr.shape != y_pos_arr.shape:
        raise ValueError("x_position and y_position must have the same shape.")

    # Define the maximum allowed coordinates for interpolation on the padded grid.
    # These are equivalent to original_shape[0] and original_shape[1] because
    # the padded grid has points from index 0 up to original_shape[dim].
    # E.g., for a 2xM original array (row indices 0, 1), the padded grid has
    # data effectively at row indices 0, 1, 2 (where 2 is the padded edge).
    # The interpolator's grid points will be [0.0, 1.0, 2.0].
    # So, max x-coordinate for interpolation is original_shape[0].
    max_x_allowed = float(original_shape[0])
    max_y_allowed = float(original_shape[1])

    if not (np.all(x_pos_arr >= 0) and np.all(y_pos_arr >= 0)):
        raise ValueError("x_position and y_position must be non-negative.")

    if np.any(x_pos_arr > max_x_allowed) or np.any(y_pos_arr > max_y_allowed):
        # Report the actual maximum input values for a more informative error message
        max_x_input = np.max(x_pos_arr) if x_pos_arr.size > 0 else -1.0
        max_y_input = np.max(y_pos_arr) if y_pos_arr.size > 0 else -1.0

        # The f-string formatting ensures at least one decimal place for the limits (e.g., "1.0" not "1")
        raise ValueError(
            f"x_position (max found: {max_x_input:.2f}) must be <= {max_x_allowed:.1f} and "
            f"y_position (max found: {max_y_input:.2f}) must be <= {max_y_allowed:.1f}."
        )

    # Pad array to replicate 'edge' behavior for boundary conditions.
    # For a 2D array, ((0, 1), (0, 1)) pads 0 before and 1 after each axis.
    # Resulting shape: (original_shape[0]+1, original_shape[1]+1)
    padded_array = np.pad(array_2d, ((0, 1), (0, 1)), mode="edge")

    # Grid points for the PADDED array.
    # These range from 0 to original_shape[0] for rows, and 0 to original_shape[1] for columns.
    # E.g., if original_shape[0] is 2, grid_rows will be [0., 1., 2.].
    grid_rows = np.arange(padded_array.shape[0], dtype=float)
    grid_cols = np.arange(padded_array.shape[1], dtype=float)

    # Create the interpolator.
    # method='linear' performs bilinear interpolation for 2D.
    # bounds_error=True will cause RGI to raise an error if query points are outside
    # the grid defined by grid_rows/grid_cols. Our explicit check above should catch this first.
    interpolator = RegularGridInterpolator(
        (grid_rows, grid_cols), padded_array, method="linear", bounds_error=True
    )

    # Prepare query points for the interpolator.
    # x_pos_arr and y_pos_arr are already relative to the original grid, which is
    # consistent with how the padded grid and its coordinates (grid_rows, grid_cols) are set up.
    if x_pos_arr.ndim == 0:  # Scalar input
        query_points = np.array([[x_pos_arr.item(), y_pos_arr.item()]])
    else:  # Array input
        query_points = np.vstack([x_pos_arr.ravel(), y_pos_arr.ravel()]).T

    interpolated_values_flat = interpolator(query_points)

    # Reshape the output to match the input x_position/y_position shape.
    if x_pos_arr.ndim == 0:
        return interpolated_values_flat[0]
    else:
        return interpolated_values_flat.reshape(x_pos_arr.shape)


# The credits for the following function go to Henry Best (https://github.com/Henry-Best-01/Amoeba)
def extract_light_curve(
    convolution_array,
    pixel_size,
    effective_transverse_velocity,
    light_curve_time_in_years,
    pixel_shift=0,
    x_start_position=None,
    y_start_position=None,
    phi_travel_direction=None,
    return_track_coords=False,
    random_seed=None,
):
    """Extracts a light curve from the convolution between two arrays by
    selecting a trajectory and calling pull_value_from_grid at each relevant
    point. If the light curve is too long, or the size of the object is too
    large, a "light curve" representing a constant magnification is returned.

    :param convolution_array: The convolution between a flux distribtion
        and the magnification array due to microlensing. Note
        coordinates on arrays have (y, x) signature.
    :param pixel_size: Physical size of a pixel in the source plane, in
        meters
    :param effective_transverse_velocity: effective transverse velocity
        in the source plane, in km / s
    :param light_curve_time_in_years: duration of the light curve to
        generate, in years
    :param pixel_shift: offset of the SMBH with respect to the convolved
        map, in pixels
    :param x_start_position: None or the x coordinate to start pulling a
        light curve from, in pixels
    :param y_start_position: None or the y coordinate to start pulling a
        light curve from, in pixels
    :param phi_travel_direction: None or the angular direction of travel
        along the convolution, in degrees
    :param return_track_coords: boolean toggle to return the x and y
        coordinates of the track in pixels
    :return: list representing the microlensing light curve
    """
    rng = np.random.default_rng(seed=random_seed)

    if isinstance(effective_transverse_velocity, Quantity):
        effective_transverse_velocity = effective_transverse_velocity.to(
            u.m / u.s
        ).value
    else:
        effective_transverse_velocity *= u.km.to(u.m)
    if isinstance(light_curve_time_in_years, Quantity):
        light_curve_time_in_years = light_curve_time_in_years.to(u.s).value
    else:
        light_curve_time_in_years *= u.yr.to(u.s)

    if pixel_shift >= np.size(convolution_array, 0) / 2:
        print(
            "warning, flux projection too large for this magnification map. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    pixels_traversed = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    )

    n_points = (
        effective_transverse_velocity * light_curve_time_in_years / pixel_size
    ) + 2

    if pixel_shift > 0:
        safe_convolution_array = convolution_array[
            pixel_shift : -pixel_shift - 1, pixel_shift : -pixel_shift - 1
        ]
    else:
        safe_convolution_array = convolution_array

    N_safe_dim_x = safe_convolution_array.shape[0]
    N_safe_dim_y = safe_convolution_array.shape[1]

    if pixels_traversed >= max(N_safe_dim_x, N_safe_dim_y):
        print(
            "Warning: light curve traversal length is too long for the safe region dimensions. Returning average flux."
        )
        return np.sum(convolution_array) / np.size(convolution_array)

    max_safe_idx_x = N_safe_dim_x - 1
    max_safe_idx_y = N_safe_dim_y - 1

    # Determine start positions
    if x_start_position is not None:
        if not (0 <= x_start_position <= max_safe_idx_x):
            print(
                f"Warning: chosen x_start_position ({x_start_position}) is outside the valid index range [0, {max_safe_idx_x}] "
                "of the safe_convolution_array. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:  # x_start_position is None, choose randomly
        if max_safe_idx_x < 0:
            print("Error: max_safe_idx_x < 0, safe_array likely misconfigured.")
            return np.sum(convolution_array) / np.size(convolution_array)

        # MODIFICATION: Choose non-border pixel if possible
        if N_safe_dim_x >= 3:  # Possible to choose non-border
            # Non-border indices are 1 to N_safe_dim_x - 2 (or max_safe_idx_x - 1)
            x_start_position = float(
                rng.integers(1, max_safe_idx_x)
            )  # low is inclusive, high is exclusive
        else:  # N_safe_dim_x is 1 or 2, all pixels are border pixels
            x_start_position = float(rng.integers(0, max_safe_idx_x + 1))

    if y_start_position is not None:
        if not (0 <= y_start_position <= max_safe_idx_y):
            print(
                f"Warning: chosen y_start_position ({y_start_position}) is outside the valid index range [0, {max_safe_idx_y}] "
                "of the safe_convolution_array. Returning average flux."
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:  # y_start_position is None, choose randomly
        if max_safe_idx_y < 0:
            print("Error: max_safe_idx_y < 0, safe_array likely misconfigured.")
            return np.sum(convolution_array) / np.size(convolution_array)

        # MODIFICATION: Choose non-border pixel if possible
        if N_safe_dim_y >= 3:  # Possible to choose non-border
            # Non-border indices are 1 to N_safe_dim_y - 2 (or max_safe_idx_y - 1)
            y_start_position = float(
                rng.integers(1, max_safe_idx_y)
            )  # low is inclusive, high is exclusive
        else:  # N_safe_dim_y is 1 or 2, all pixels are border pixels
            y_start_position = float(rng.integers(0, max_safe_idx_y + 1))

    if phi_travel_direction is not None:
        angle = phi_travel_direction * np.pi / 180
        delta_x = pixels_traversed * np.cos(angle)
        delta_y = pixels_traversed * np.sin(angle)

        if (
            x_start_position + delta_x >= np.size(safe_convolution_array, 0)
            or y_start_position + delta_y >= np.size(safe_convolution_array, 1)
            or x_start_position + delta_x < 0
            or y_start_position + delta_y < 0
        ):
            print(
                "Warning, chosen track leaves the convolution array. Returning average flux.",
                f"x_start_position: {x_start_position}, y_start_position: {y_start_position}, "
                f"delta_x: {delta_x}, delta_y: {delta_y}",
                f"x_end_position: {x_start_position + delta_x}, "
                f"y_end_position: {y_start_position + delta_y}",
            )
            return np.sum(convolution_array) / np.size(convolution_array)
    else:
        success = None
        backup_counter = 0
        angle = rng.random() * 360 * np.pi / 180
        while success is None:
            angle += np.pi / 2
            delta_x = pixels_traversed * np.cos(angle)
            delta_y = pixels_traversed * np.sin(angle)
            if (
                x_start_position + delta_x < np.size(safe_convolution_array, 0)
                and y_start_position + delta_y < np.size(safe_convolution_array, 1)
                and x_start_position + delta_x >= 0
                and y_start_position + delta_y >= 0
            ):
                success = True
            backup_counter += 1
            if backup_counter > 4:  # pragma: no cover
                break

    x_positions = np.linspace(
        x_start_position, x_start_position + delta_x, 5 * int(n_points)
    )
    y_positions = np.linspace(
        y_start_position, y_start_position + delta_y, 5 * int(n_points)
    )

    light_curve = pull_value_from_grid(safe_convolution_array, x_positions, y_positions)

    if return_track_coords:
        return (
            np.asarray(light_curve),
            x_positions + pixel_shift,
            y_positions + pixel_shift,
        )

    return np.asarray(light_curve)


# Credits: Luke Weisenbach (https://github.com/weisluke/microlensing/blob/main/microlensing/Util/length_scales.py)
def theta_star_physical(
    z_lens: float,
    z_src: float,
    cosmo: Cosmology,
    m: float = 1,
) -> tuple:
    """Calculate the size of the Einstein radius of a point mass lens in the
    lens and source planes, in meters.

    :param z_lens: lens redshift
    :param z_src: source redshift
    :param cosmo: an astropy.cosmology instance.
    :param m: point mass lens mass in solar mass units
    :return theta_star: theta_star in the lens plane in arcseconds
    :return theta_star_lens: theta_star in the lens plane in meters
    :return theta_star_src: theta_star in the source plane in meters
    """
    microlens_mass = m * u.M_sun

    D_d = cosmo.angular_diameter_distance(z_lens)
    D_s = cosmo.angular_diameter_distance(z_src)
    D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_src)

    theta_star = (
        np.sqrt(4 * const.G * microlens_mass / const.c**2 * D_ds / (D_s * D_d)) * u.rad
    )
    theta_star_lens = theta_star.to(u.rad).value * D_d
    theta_star_src = theta_star_lens * D_s / D_d

    return theta_star.to(u.arcsec), theta_star_lens.to(u.m), theta_star_src.to(u.m)


def get_tau_sf_from_distribution_agn_variability(
    black_hole_mass_exponent, known_mag_abs, z_src, means, cov, nsamps=1
):
    """Draw Tau and SF_inf from the joint distribution conditioned on the BH
    mass, absolute magnitude, and source redshift.

    The joint distribution is a multivariate normal distribution in
    log(BH_mass/Msun), known_mag_abs, log_(SF_inf/mag), log_(tau/days),
    zsrc space.

    :param black_hole_mass_exponent: log_{10} of the black hole mass in
        solar masses.
    :param known_mag_abs: Absolute magnitude of the point source in some
        known band.
    :param z_src: Redshift of the source.
    :param means: List of means of the joint distribution. Order:
        log(BH_mass/Msun), known_mag_abs, log_(SF_inf/mag),
        log_(tau/days), zsrc
    :param cov: Covariance matrix of the joint distribution. Shape (5,
        5).
    :param nsamps: Number of samples to draw from the joint
        distribution.
    :return: SF_inf, tau drawn from the conditional distribution.
        Returns a tuple of numpy arrays (SF_inf, tau). If nsamps == 1,
        returns a tuple of scalars.
    """
    means = np.array(means)
    cov = np.array(cov)

    # 1. Define indices based on the docstring order:
    # 0: log(BH_mass)  [Given]
    # 1: known_mag_abs [Given]
    # 2: log(SF_inf)   [Target]
    # 3: log(tau)      [Target]
    # 4: zsrc          [Given]

    idx_given = [0, 1, 4]  # The variables we observe
    idx_target = [2, 3]  # The variables we want to sample

    # 2. Partition the Mean Vector
    mu_given = means[idx_given]
    mu_target = means[idx_target]

    # 3. Partition the Covariance Matrix
    # Covariance of the targets (2x2)
    cov_target_target = cov[np.ix_(idx_target, idx_target)]
    # Covariance of the given variables (3x3)
    cov_given_given = cov[np.ix_(idx_given, idx_given)]
    # Cross-covariance (2x3)
    cov_target_given = cov[np.ix_(idx_target, idx_given)]

    # Calculate Conditional Mean and Covariance
    # solve for x in: cov_given_given * x = (observed_values - mu_given)
    observed_values = np.array([black_hole_mass_exponent, known_mag_abs, z_src])
    diff = observed_values - mu_given

    # mu_cond = mu_target + cov_target_given * cov_given_given^-1 * diff
    term1 = np.linalg.solve(cov_given_given, diff)
    cond_mean = mu_target + np.dot(cov_target_given, term1)

    # cov_cond = cov_target_target - cov_target_given * cov_given_given^-1 * cov_given_target
    # cov_given_target is the transpose of cov_target_given
    term2 = np.linalg.solve(cov_given_given, cov_target_given.T)
    cond_cov = cov_target_target - np.dot(cov_target_given, term2)

    # Sample from the new conditional distribution
    # This returns shape (nsamps, 2) where col 0 is SF and col 1 is Tau
    samples = np.random.multivariate_normal(cond_mean, cond_cov, size=nsamps)

    sf_inf_samples = samples[:, 0]
    tau_samples = samples[:, 1]

    if nsamps == 1:
        return sf_inf_samples[0], tau_samples[0]

    return sf_inf_samples, tau_samples


def get_breakpoint_frequency_and_std_agn_variability(log_SF_inf, log_tau):
    """Convert SF_inf and tau to breakpoint frequency and standard deviation.

    :param log_SF_inf: log_{10} of SF_inf in magnitudes.
    :param log_tau: log_{10} of tau in days.
    :return: log_{10} of the breakpoint frequency in 1/days and standard
        deviation in magnitudes.
    """
    SF_inf = 10**log_SF_inf  # in mag
    tau = 10**log_tau  # in days

    standard_deviation = SF_inf / np.sqrt(2)  # in mag
    breakpoint_frequency = 1 / (2 * np.pi * tau)  # in 1/days
    log_breakpoint_frequency = np.log10(breakpoint_frequency)

    return log_breakpoint_frequency, standard_deviation
