#!/usr/bin/env python
import numpy as np
import kcorrect.kcorrect
from uncertainties import unumpy


def kcorr_sdss(
    mags_sdss,
    redshift,
    responses=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
    band_shift=0.0,
):
    """Computes the astronomical K correction for galaxies on the SDSS
    broadband filters using the kcorrect module based on Blanton and Roweis
    2007.

    :param mags_sdss: numpy.ndarray The multi-band SDSS magnitudes of
        all the targets. A 2D array of uncertainties.core.Variable, with
        bands along the rows and targets along the columns. If your
        magnitude is m1 and associated error is e1,
        uncertainties.core.Variable should be: ufloat(m1, e1).
    :param redshift: numpy.ndarray An array of the redshifts of the
        deflectors. A 1D array of floats.
    :param responses: list of str The SDSS bands for which the magnitude
        is provided.
    :param responses_out: list of str The SDSS bands on which you want
        the K-corrections.
    :return: numpy.ndarray A 2-D array with K-correction for all the
        targets, with each row containing the K-correction for output
        bands for each target.
    """

    # Extract the magnitudes and errors in separate arrays.
    mags = unumpy.nominal_values(mags_sdss).T
    mag_errs = unumpy.std_devs(mags_sdss).T

    kc = kcorrect.kcorrect.Kcorrect(responses=responses)

    maggies_ivar = np.zeros(mag_errs.shape, dtype=np.float32)
    maggies = np.zeros(mags.shape, dtype=np.float32)

    mag_low = mags - mag_errs
    mag_high = mags + mag_errs

    for j in range(len(maggies)):
        for k in np.arange(len(responses), dtype=int):
            maggies[j, k] = 10 ** (-0.4 * mags[j, k])
            maggies_ivar[j, k] = 0.5 * (
                10 ** (-0.4 * mag_low[j, k]) - 10 ** (-0.4 * mag_high[j, k])
            )

    coeffs = kc.fit_coeffs(redshift=redshift, maggies=maggies, ivar=maggies_ivar)
    k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=band_shift)

    # return the k-correction coefficients for the provided targets and bands in a 2D array of floats.
    return k
