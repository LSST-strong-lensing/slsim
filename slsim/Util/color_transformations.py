#!/usr/bin/env python


def LSST_to_SDSS(lsst_u, lsst_g, lsst_r, lsst_i, lsst_z):
    """Converts the five bands LSST magnitudes to the SDSS magnitudes using
    empirically derived transformations. These transformations are only valid
    for red galaxies and should not be used for blue galaxies or other stellar
    populations. The coefficients are based on polynomial fits to color terms.

    :param lsst_u: u-band LSST magnitude
    :type lsst_u: float
    :param lsst_g: g-band LSST magnitude
    :type lsst_g: float
    :param lsst_r: r-band LSST magnitude
    :type lsst_r: float
    :param lsst_i: i-band LSST magnitude
    :type lsst_i: float
    :param lsst_z: z-band LSST magnitude
    :type lsst_z: float
    :return: A list of SDSS-equivalent magnitudes in the order [u, g, r,
        i, z].
    :rtype: list
    """
    sdss_u = (
        lsst_u
        - 0.014285
        + (0.191787 * (lsst_u - lsst_g))
        + (-0.062736 * (lsst_u - lsst_g) ** 2)
    )
    sdss_g = (
        lsst_g
        + 0.008059
        + (0.029470 * (lsst_g - lsst_r))
        + (0.031589 * (lsst_g - lsst_r) ** 2)
    )
    sdss_r = (
        lsst_r
        - 0.001168
        + (0.017418 * (lsst_r - lsst_i))
        + (0.021144 * (lsst_r - lsst_i) ** 2)
    )
    sdss_i = (
        lsst_i
        - 0.000026
        + (0.044532 * (lsst_i - lsst_z))
        + (-0.013802 * (lsst_i - lsst_z) ** 2)
    )
    sdss_z = (
        lsst_z
        - 0.030518
        + (-0.206242 * (lsst_i - lsst_z))
        + (0.084968 * (lsst_i - lsst_z) ** 2)
    )

    return [sdss_u, sdss_g, sdss_r, sdss_i, sdss_z]
