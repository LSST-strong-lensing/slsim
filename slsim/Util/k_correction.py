#!/usr/bin/env python
import numpy as np
import kcorrect.kcorrect
from uncertainties import unumpy

"""
This module provides function to calculate the astronomical k-correction for the galaxies
using the kcorrect module based on Blanton and Roweis 2007.
"""


def kcorr_sdss(
    mags_sdss,
    redshift,
    responses=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
    responses_out=["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"],
    band_shift=0.0,
    redshift_range=[0, 2],
):
    """Computes the astronomical K correction for galaxies on the SDSS broadband filters
    using the kcorrect module based on Blanton and Roweis 2007.

    input_params:
    mags_sdss : The multi-band SDSS magnitudes of all the targets.
    type      : a 2D array of uncertainties.core.Variable, with bands along the
                rows and targets along the column.
            If your magnitude is m1 and associated error is e1,
            uncertainties.core.Variable should be : ufloat(m1,e1).

    redshift:   an array of the redshifts of the deflectors
    type:       a 1D array fo floats.

    responses: the sdss bands for which the magnitude is provided
    type:      an array of strings

    responses_out: the sdss bands on which you want the k-corrections
    type:      an array of strings

    returns:
    a 2-D array with K-correction for all the targets with each row containing
    the k-correction for output bands for each target.
    """

    # Etract the magnitudes and errors in separate arrays.
    mags = unumpy.nominal_values(mags_sdss).T
    mag_errs = unumpy.std_devs(mags_sdss).T

    all_bands = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
    all_b0 = np.array([1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10], dtype=np.float32)
    # coefficients taken from kcorrect module
    # https://github.com/blanton144/kcorrect/blob/main/src/kcorrect/utils.py

    # Use the coefficients for only the available bands
    indices = [all_bands.index(band) for band in responses]
    b0 = all_b0[indices]

    kc = kcorrect.kcorrect.Kcorrect(
        responses=responses, responses_out=responses_out, redshift_range=redshift_range
    )

    # convert the magnitudes and errors to maggies and inverse_maggies
    maggies_ivar = np.zeros(mag_errs.shape, dtype=np.float32)
    maggies = np.zeros(mags.shape, dtype=np.float32)

    for j in range(len(maggies)):
        for k in np.arange(len(responses), dtype=int):
            maggies[j, k] = (
                2.0
                * b0[k]
                * np.sinh(-np.log(b0[k]) - (0.4 * np.log(10.0) * mags[j, k]))
            )
            maggies_err = (
                2.0
                * b0[k]
                * np.cosh(-np.log(b0[k]) - (0.4 * np.log(10.0) * mags[j, k]))
                * 0.4
                * np.log10(10.0)
                * mag_errs[j, k]
            )
            maggies_ivar[j, k] = 1.0 / maggies_err**2

    coeffs = kc.fit_coeffs(redshift=redshift, maggies=maggies, ivar=maggies_ivar)
    k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=band_shift)

    # return the k-correction coefficients for the provided targets and bands in a 2D array of floats.
    return k
