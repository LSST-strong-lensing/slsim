#!/usr/bin/env python
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy
from astropy.modeling.models import Linear1D
from slsim.Util.k_correction import kcorr_sdss
from astropy.cosmology import FlatLambdaCDM
from slsim.Util.color_transformations import LSST_to_SDSS

"""
This module provides function to calculate the central stellar velocity dispersion of the deflector 
(elliptical galaxies) using LSST broadband magnitudes and the redshift. It assumes the evolution of 
galaxy luminosity function, as discussed in Bell et al 2004, Blanton et al 2003, and uses the 
scaling relations for L-sigma relationship from Choi et al 2007 (derived from spectroscopic 
measurements) and Parker et al 2007 (derived from weak lensing measurements). The user has the 
option to decide which scaling relations he/she wants to use, the one derived from spectroscopic
measurements or from the weak lensing measurements.
"""


def Lsigma_relation_spectroscopic(mgSDSS, mrSDSS, Dlum, redshift):
    """
    input params:

    mgSDSS: k-corrected g-band magnitude of the deflector
    type:   a 1D array of floats

    mrSDSS: k-corrected r-band magnitude of the deflector
    type:   a 1D array of floats

    Dlum: distance luminosity of the deflector
    type: a 1D array of floats

    redshift: redshift of the deflector
    type: a 1D array of floats

    .. [1] Bell et al., (2004), astro-ph/0303394, doi: 10.1086/420778
    .. [2] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060
    """

    # Use the SDSS g-band and r-band magnitudes to get the B-band apparent magnitude of the galaxy using the relation
    # given in equation A2, Appendix, Bell et al 2004 for red galaxies. This is required only for using the relations
    # based on spectroscopic measurements.
    MabsB = mgSDSS + 0.155 + 0.370 * (mgSDSS - mrSDSS)

    # Convert the apparent B-band magnitude to the absolute B-band magnitude using the redshift and cosmology defined
    MabsB = MabsB - 5.0 * np.log10(Dlum / 10)
    """Now using the data from DEEP2 and COMBO-17 surveys, Bell et 2004 found
    that the B-band luminosity function evolves such that characteristic
    magnitude MBstar decline by 1.5 magnitudes from z=0.0 to z=1.0. We use the
    same assumption here;

    Hence, MBstar and redshift should follow the relation, i.e., MBstar
    = MBstar0-(redshift)*1.5. where MBstar0 = MBstar(at redshift=0). In
    our case, MBstar0 = -19.31 has been estimated from the mean value of
    the MBstar0, from Table 1, Bell et al 2004.
    """

    # define a 1D line model for MBstar evolution with redshift.
    MBstar_func = Linear1D(-1.5, -19.31)

    # Use the above model to calculate MB* at the deflector redshift
    MBstar = MBstar_func(redshift)

    # Calculate L/L* using the magnitude-luminosity relation
    LbyLstar = 10.0 ** (-0.4 * (MabsB - MBstar))
    """Now use the L-sigma relation for the elliptical galaxies i.e., the Faber
    Jackson relation, sigma/sigma_star = (L/Lstar)**(1/alpha) and taking the
    sigma* and alpha value from Choi et al 2007, derived for early type
    galaxies, calculate the the velocity dispersion sigma."""
    sigma_star, alpha = ufloat(161, 5), 2.32  # Choi et al 2007

    # Use sigma_star and alpha values to calculate the stellar velocity dispersion sigma
    sigma = sigma_star * LbyLstar ** (1 / alpha)

    return sigma


def Lsigma_relation_weaklensing(mrSDSS, miSDSS, Dlum, redshift):
    """
    input params:

    mrSDSS: k-corrected r-band magnitude of the deflector
    type:   a 1D array of floats

    miSDSS: k-corrected i-band magnitude of the deflector
    type:   a 1D array of floats

    Dlum: distance luminosity of the deflector
    type: a 1D array of floats

    redshift: redshift of the deflector
    type: a 1D array of floats

    .. [1] Blanton et al., (2003), astro-ph/0210215, doi: 10.1086/375776
    .. [2] Parker et al., (2007),  arXiv:0707.1698, doi: 10.1086/521541
    """

    # Convert the apparent r-band magnitudes to the absolute r
    Mabsr = mrSDSS - 5.0 * np.log10(Dlum / 10)

    # Convert the sdss r-mag to r'-mag from Frei & Gunn 2003 (Table 3).
    # r' is a fake filter i.e., r shifted to z=0.1.
    Mabsr = Mabsr - 0.11
    """We assume the same assumption here (from Bell et al 2004) for decline of
    characteristic magnitude Mrstar for r'-band,

    Hence, Mrstar and redshift should follow the relation, i.e., Mrstar
    = Mrstar0-(redshift-0.1)*1.5. where Mrstar0 = Mrstar(at
    redshift=0.1).

    In our case, Mrstar0 = -20.44 has been estimated from Table 2,
    Blanton et al 2003.
    """

    Mrstar0 = -20.44  # calculated at redhift=0.1
    # Use the above value to calculate Mrstar at the deflector redshift
    Mrstar = Mrstar0 - (redshift - 0.1) * 1.5

    # Calculate L/L* using the magnitude-luminosity relation
    LbyLstar = 10.0 ** (-0.4 * (Mabsr - Mrstar))
    """Now use the L-sigma relation and taking the sigma_star and alpha value
    from Parker et al 2007, derived using weak-lensing measurements, calculate
    the the velocity dispersion sigma."""
    # sigma_star, alpha = 142+-18, 3      # Parker et al 2007

    sigma_star_nominal = np.ones(len(LbyLstar)) * 142
    sigma_star_stdev = np.ones(len(LbyLstar)) * 18
    alpha = np.ones(len(LbyLstar)) * 3
    sigma_star = unumpy.uarray(sigma_star_nominal, sigma_star_stdev)
    sigma_star[miSDSS > 20.5] = ufloat(137, 11)
    alpha[miSDSS > 20.5] = 3

    # Use sigma_star and alpha values to calculate the stellar velocity dispersion sigma
    sigma = sigma_star * LbyLstar ** (1 / alpha)

    return sigma


def get_velocity_dispersion(
    deflector_type,
    lsst_mags,
    lsst_errs,
    redshift,
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
    bands=["u", "g", "r", "i", "z"],
    scaling_relation="spectroscopic",
):
    """
    input_params:

    deflector_type: type of the foreground/ deflector, e.g., 'elliptical'
    type: string

    lsst_mags: a 2D array of the lsst magnitudes of the deflector with multi-band magnitudes
    along the row, and different deflector along the column.
    type: a 2D array of floats.

    lsst_errs: a 2D array of the lsst magnitude errors of the deflector with multi-band errors
    along the row, and different deflector along the column.
    type: a 2D array of floats.

    redshift:   a 1D array of the redshifts
    type: a 1D array of floats

    cosmo: cosmology defined
    type: astropy.cosmology

    bands: bands for which you're providing the magnitudes,
    type: a list of strings e.g., ['u', 'g', 'r', 'i', 'z' ]


    returns:
    stellar velocity dispersion [km/s]

    References
    ----------
    .. [1] Bell et al., (2004), astro-ph/0303394, doi: 10.1086/420778
    .. [2] Blanton & Roweis (2007), astro-ph/0606170, doi: 10.1086/510127
            https://kcorrect.readthedocs.io/en/5.1.2/
    .. [3] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060
    .. [4] Blanton et al., (2003), astro-ph/0210215, doi: 10.1086/375776
    .. [5] Parker et al., (2007),  arXiv:0707.1698, doi: 10.1086/521541
    """

    if deflector_type != "elliptical":
        raise KeyError("The module currently supports only elliptical galaxies.")

    muSDSS, mgSDSS, mrSDSS, miSDSS, mzSDSS = LSST_to_SDSS(
        unumpy.uarray(lsst_mags[0], lsst_errs[0]),
        unumpy.uarray(lsst_mags[1], lsst_errs[1]),
        unumpy.uarray(lsst_mags[2], lsst_errs[2]),
        unumpy.uarray(lsst_mags[3], lsst_errs[3]),
        unumpy.uarray(lsst_mags[4], lsst_errs[4]),
    )

    if scaling_relation == "spectroscopic":
        # for k-correction upto redshift z=0 only
        band_shift = 0.0

    elif scaling_relation == "weak-lensing":
        # for k-correction upto redshift z=0.1
        # since the scaling relations used are at z=0.1
        band_shift = 0.1

    # Find out the K-correction factor using the kcorrect module by Blanton
    k_corrections = kcorr_sdss(
        np.array([muSDSS, mgSDSS, mrSDSS, miSDSS, mzSDSS]),
        redshift,
        band_shift=band_shift,
    )

    # Apply the K-correction on the SDSS magnitudes
    muSDSS = muSDSS - k_corrections[:, 0]
    mgSDSS = mgSDSS - k_corrections[:, 1]
    mrSDSS = mrSDSS - k_corrections[:, 2]
    miSDSS = miSDSS - k_corrections[:, 3]
    mzSDSS = mzSDSS - k_corrections[:, 4]

    ## Note: It will be better if we apply the K-correction directly on the LSST magnitudes,
    ## but no such relation is known to Vibhore right now.

    # calculates the distance luminosity using the redshift and the cosmology in 'pc'
    Dlum = cosmo.luminosity_distance(redshift).to("pc").value

    if scaling_relation == "spectroscopic":
        # Use the Lsigma relation based on spectroscopic measurements to calculate the
        # sigma of the deflector
        sigma = Lsigma_relation_spectroscopic(mgSDSS, mrSDSS, Dlum, redshift)

    elif scaling_relation == "weak-lensing":
        # Use the Lsigma relation based on weak-lensing measurements to calculate the
        # sigma of the deflector
        sigma = Lsigma_relation_weaklensing(mrSDSS, miSDSS, Dlum, redshift)

    # returns the calculated sigma
    # type: a 1D array of uncertainties.core.Variable
    ##   to extract the nomianl values and the uncertainities in separate arrays,
    ##   use unumpy.nominal_values(sigma) and unumpy.std_devs(sigma)

    return sigma
