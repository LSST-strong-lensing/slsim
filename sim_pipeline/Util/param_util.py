import numpy as np


def epsilon2e(epsilon):
    """
    Translates ellipticity definitions from

    .. math::
        epsilon = \\equic \\frac{1 - q^2}{1 + q^2}

    to

    .. math::
        e = \\equic \\frac{1 - q}{1 + q}

    :param epsilon: ellipticity
    :return: eccentricity
    """
    if epsilon == 0:
        return 0
    elif 0 < epsilon <= 1:
        return (1 - np.sqrt(1 - epsilon**2)) / epsilon
    else:
        raise ValueError('Value of "epsilon" is %s and needs to be in [0, 1]' % epsilon)


def e2epsilon(e):
    """

    translates ellipticity definitions from

    .. math::
        e = \\equic \\frac{1 - q}{1 + q}

    to

    .. math::
        epsilon = \\equic \\frac{1 - q^2}{1 + q^2}

    :param e: eccentricity
    :return: ellipticity
    """
    return 2 * e / (1 + e**2)


def random_ra_dec(ra_min, ra_max, dec_min, dec_max, n):
    """
    Generates n number of random ra, dec pair with in a given limits.
    
    :param ra_min: minimum limit for ra
    :param ra_max: maximum limit for ra
    :param dec_min: minimum limit for dec
    :param dec_max: maximum limit for dec
    :param n: number of random sample
    :returns: n number of ra, dec pair within given limits
    """
    ra=np.random.uniform(ra_min,ra_max, n)
    dec=np.random.uniform(dec_min, dec_max, n)
    return ra, dec
