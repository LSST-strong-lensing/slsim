import numpy as np
from astropy import constants as const
from astropy import units as u


def spin_to_isco(spin):
    """Converts dimensionless spin parameter of a black hole to the innermost stable
    circular orbit in gravitational radii [R_g = GM/c^2, with units length]

    :param spin: Dimensionless spin of black hole, ranging from -1 to 1. Positive values
        represent orbits aligned with the black hole spin.
    :return: float value of innermost stable circular orbit, ranging from 1 to 9.
    """
    if abs(spin) > 1:
        raise ValueError("Absolute value of spin cannot exceed 1")
    # Calculate intermediate values
    z1 = 1 + (1 - spin**2) ** (1 / 3) * (
        (1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3)
    )
    z2 = (3 * spin**2 + z1**2) ** (1 / 2)

    # Return ISCO distance in gravitational radii
    return 3 + z2 - np.sign(spin) * ((3 - z1) * (3 + z1 + 2 * z2)) ** (1 / 2)


def calculate_eddington_luminosity(black_hole_mass_exponent):
    """Calculates the Eddington luminosity for a black hole mass exponent.

    Eddington_luminosity = 4 * pi * G * black_hole_mass * mass_proton
                              * c / sigma_thompson

    :param black_hole_mass_exponent: log of the black hole mass normalized by the mass of the sun.
        - black_hole_mass_exponent = log_10(black_hole_mass / mass_sun)
        Typical AGN have an exponent ranging from 6 to 10.
    :return: Eddington luminosity
    """
    black_hole_mass = convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent)
    return 4 * np.pi * const.G * black_hole_mass * const.m_p * const.c / const.sigma_T


def eddington_ratio_to_accretion_rate(
    black_hole_mass_exponent, eddington_ratio, efficiency=0.1
):
    """Calculates the mass that must be accreted by the accretion disk.

    for the accretion disk to radiate at the desired Eddington ratio.
    Bolometric_luminosity = mass_accreted * c^2 * efficiency

    :param black_hole_mass_exponent: log of the black hole mass normalized by the mass
        of the sun. black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param eddington_ratio: desired Eddington ratio defined as a fraction of bolometric
        luminosity / Eddington luminosity.
    :param efficiency: the efficiency of mass-to-energy conversion in accretion disk
    :return: required mass_accreted for accretion disk to radiate at the desired
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
    """Calculates the gravitational radius (r_g) of a black hole. The gravitational.

    radius defines the typical size scales around a black hole for AGN.
    The formula for gravitational radius is: r_g  = G * mass / c^2

    :param black_hole_mass_exponent: log of the black hole mass normalized by the mass
        of the sun. black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :return: the gravitational radius in astropy length units
    """
    black_hole_mass = convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent)
    return const.G * black_hole_mass / const.c**2


def convert_black_hole_mass_exponent_to_mass(black_hole_mass_exponent):
    """This function takes in the log of the black hole mass normalized by the mass of
    the sun and returns the mass of the black hole.

    :param black_hole_mass_exponent: log of the black hole mass normalized by the mass
        of the sun. black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :return: mass of the black hole in astropy units
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
    :param black_hole_spin: the dimensionless spin parameter of the black hole, where
        the spinless case (spin = 0) corresponds to a Schwarzschild black hole.
        Positive spin represents the accretion disk's angular momentum is aligned
        with the black hole's spin, and negative spin represents retrograde accretion
        flow.
    :param black_hole_mass_exponent: log of the black hole mass normalized by the mass
        of the sun. black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param eddington_ratio: the fraction of the eddington limit which the black hole
        is accreting at.
    :return: the temperature of the accretion disk at all radii in units Kelvin.
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

    :param temperature: Temperature of the black body, in Kelvin
    :param wavelength_in_nanometers: Emitted wavelength in local rest frame in nanometers
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

    :param temperature: Temperature of the black body, in Kelvin
    :param wavelength_in_nanometers: Emitted wavelength in local rest frame in nanometers
    :return: The derivative of the spectral radiance with respect to temperature for a black body
    """

    return (
        planck_law(temperature + 1e-2, wavelength_in_nanometers)
        - planck_law(temperature, wavelength_in_nanometers)
    ) / 1e-2


def create_radial_map(r_out, r_resolution, inclination_angle):
    """This creates a 2-dimentional array of radial positions where the maximum radius
    is defined by r_out, and the radial resolution is defined by r_resolution.

    :param r_out: maximum radial value. For an accretion disk, this can be 10^3 to 10^5.
    :param r_resolution: the number of points between r = 0 and r = r_out. The final map
        will be shape (2 * r_resolution), (2 * r_resolution)
    :param inclination_angle: the inclination of the plane of the accretion disk with
        respect to the observer in degrees.
    :return: a 2-dimensional array of radial positions of shape ((2 * r_resolution), (2
        * r_resolution)) in the projected plane of the sky
    """
    x_values = np.linspace(-r_out, r_out, 2 * r_resolution)
    y_values = np.linspace(-r_out, r_out, 2 * r_resolution) / np.cos(
        inclination_angle * np.pi / 180
    )

    X, Y = np.meshgrid(x_values, y_values, indexing="ij")

    return (X**2.0 + Y**2.0) ** 0.5


def create_phi_map(r_out, r_resolution, inclination_angle):
    """This creates a 2-dimentional array of phi values at all positions where the
    maximum radius is defined by r_out, and the radial resolution is defined by
    r_resolution.

    :param r_out: maximum radial value. For an accretion disk, this can be 10^3 to 10^5.
    :param r_resolution: the number of points between r = 0 and r = r_out. The final map
        will be shape (2 * r_resolution), (2 * r_resolution)
    :param inclination_angle: the inclination of the plane of the accretion disk with
        respect to the observer in degrees.
    :return: a 2-dimensional array of phi values at radial positions of shape ((2 *
        r_resolution), (2 * r_resolution)) in the projected plane of the sky, such that
        phi = 0 is nearest to the observer.
    """
    x_values = np.linspace(-r_out, r_out, 2 * r_resolution)
    y_values = np.linspace(-r_out, r_out, 2 * r_resolution) / np.cos(
        inclination_angle * np.pi / 180
    )

    X, Y = np.meshgrid(x_values, y_values, indexing="ij")
    # must add pi/2 so phi = 0 points towards observer.
    return (np.arctan2(Y, X) + np.pi / 2) % (2 * np.pi)


def calculate_time_delays_on_disk(
    r_out, r_resolution, inclination_angle, black_hole_mass_exponent, corona_height
):
    """This calculates the light travel time lags from a point source corona located
    above the black hole to simulate the lamppost geometry. The corona is defined as the
    point-souce approximation of the X-ray variable source.

    The light travel time lags, tau(r, phi), are defined in the lamppost geometry through:

    c * tau(r, phi|h_corona, inclination) = sqrt(h_corona^2 + r^2) +
                                    h_corona * cos(inclination) - r * sin(inclination) * cos(phi)

    :param r_out: maximum radial value. For an accretion disk, this can be 10^3 to 10^5.
    :param r_resolution: the number of points between r = 0 and r = r_out. The final map will
        be shape (2 * r_resolution), (2 * r_resolution)
    :param inclination_angle: the tilt of the accretion disk with respect to the observer.
        Zero degrees is face on, 90 degrees is edge on.
    :param black_hole_mass_exponent: log of the black hole mass normalized by the mass
        of the sun. black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param corona_height: height of the corona in gravitational_radii. Typical values range
        from 10 to 100 gravitational radii.
    :return: A 2-dimensional array of time delays between the corona and the accretion disk.
    """

    gravitational_radius = calculate_gravitational_radius(black_hole_mass_exponent)

    radial_map = create_radial_map(r_out, r_resolution, inclination_angle)
    phi_map = create_phi_map(r_out, r_resolution, inclination_angle)

    return (gravitational_radius / const.c) * (
        (radial_map**2.0 + corona_height**2.0) ** 0.5
        + corona_height * np.cos(inclination_angle)
        - radial_map * np.cos(phi_map) * np.sin(inclination_angle)
    )
