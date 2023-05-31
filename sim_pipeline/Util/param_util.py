import numpy as np


def epsilon2e(epsilon):
    """

    translates ellipticity definitions from

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