import numpy as np

from typing import Union


def vel_disp_from_m_star(m_star: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the velocity dispersion of a galaxy from its stellar mass using an
    empirical power-law relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::
        V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
        M_\\odot} \\right)^{0.24}

    Values taken from table 2 of [1]

    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    Args:
        m_star (Union[float, np.ndarray]): Stellar mass of the galaxy in solar masses.

    Returns:
        Union[float, np.ndarray]: Velocity dispersion in km/s.
    """

    v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)

    return v_disp
