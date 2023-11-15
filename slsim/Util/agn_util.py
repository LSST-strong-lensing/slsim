import numpy as np
from astropy import constants as const


def spin_to_isco(spin):
    """Converts dimensionless spin parameter of a black hole to the innermost stable
    circular orbit (in gravitational radii) [R_g = GM/c^2]

    :param spin: Dimensionless spin of black hole, ranging from -1 to 1.
    :return: value of innermost stable circular orbit, ranging from 1 to 9.
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


def calculate_eddington_luminosity(mass_bh_exponent):
    """Calculates the Eddington luminosity for a black hole mass exponent.

    Eddington_luminosity = 4 * pi * G * mass_bh * mass_proton
                              * c / sigma_thompson

    :param mass_bh_exponent: solution to log_10(mass_bh / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :return: Eddington luminosity
    """
    mass_bh = 10**mass_bh_exponent * const.M_sun
    return 4 * np.pi * const.G * mass_bh * const.m_p * const.c / const.sigma_T


def eddington_ratio_to_accreted_mass(mass_bh_exponent, eddington_ratio, efficiency=0.1):
    """Calculates the mass that must be accreted by the accretion disk.

    for the accretion disk to radiate at the desired Eddington ratio.
    Bolometric_luminosity = mass_accreted * c^2 * efficiency

    :param mass_bh_exponent: solution to log_10(mass_bh / mass_sun). Typical AGN have an
        exponent ranging from 6 to 10.
    :param eddington_ratio: desired Eddington ratio defined as a fraction of bolometric
        luminosity / Eddington luminosity.
    :param efficiency: the efficiency of mass-to-energy conversion in accretion disk
    :return: required mass_accreted for accretion disk to radiate at the desired
        Eddington ratio
    """
    if efficiency <= 0:
        raise ValueError("Efficiency cannot be negative")

    # Calculate Eddington luminosity
    l_eddington = calculate_eddington_luminosity(mass_bh_exponent)

    # Calculate required accreted mass to reach Eddington luminosity
    m_eddington_accreted = l_eddington / (efficiency * const.c**2)

    return eddington_ratio * m_eddington_accreted
