#!/usr/bin/env python
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy
from astropy.modeling.models import Linear1D
from slsim.Util.k_correction import kcorr_sdss
from astropy.cosmology import FlatLambdaCDM

"""
This module provides function to calculate the central stellar velocity dispersion of the deflector 
(elliptical galaxies) using LSST broadband magnitudes and the redshift. It assumes the evolution of 
galaxy luminoisity function, as discussed in Bell et al 2004, Blanton et al 2003, and uses the 
scaling relations for L-sigma relationship from Choi et al 2007 (derived from spectroscopic 
measurements) and Parker et al 2007 (derived from weak lensing measurements). The user has the 
option to decide which scaling relations he/she wants to use, the one derived from spectroscopic
measurements or from the weak lensing measurements.
"""


def get_velocity_dispersion(
    deflector_type,
    lsst_mags,
    lsst_errs,
    redshift,
    cosmo=FlatLambdaCDM(H0=72, Om0=0.26),
    bands=["g", "r", "i"],
    c1=0.01011,
    c2=0.01920,
    c3=0.05162,
    c4=-0.00032,
    c5=0.06555,
    c6=-0.02949,
    c7=0.00003,
    c8=0.04040,
    c9=-0.00892,
    c10=-0.03068,
    c11=-0.21527,
    c12=0.09394,
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

    Note: Please provide atleast three bands data, including the g, r, and i bands.
    The three bands are required to perform the k-correction in the SDSS bands. If there is some other
    way of doing k-correction directly in the LSST bands, we will need only the two g and r bands data.

    redshift:   a 1D array of the redshifts
    type: a 1D array of floats

    cosmo: cosmology defined
    type: astropy.cosmology.flrw.lambdacdm.FlatLambdaCDM

    bands: bands for which you're providing the magnitudes, for now use only 'g', 'r', and 'i'
    type: a list of strings e.g., ['g','r','i']

    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12: The color conversion coefficients to convert the LSST magnitudes
    to the SDSS magnitudes as per the equations (derived from the red galaxy catalog)
        ##  sdss_g  = lsst_g + c0 + (c1*(lsst_g-lsst_r))   + (c2*(lsst_g-lsst_r)**2)
        ##  sdss_r  = lsst_r + c3 + (c4*(lsst_g-lsst_r))   + (c5*(lsst_g-lsst_r)**2)
        ##  sdss_i  = lsst_i + c7 +  (c8*(lsst_i-lsst_z))  +  (c9*(lsst_i-lsst_z)**2)
        ########    if the z-band magnitude is not available:
        ########    sdss_i  = lsst_i + c7 +  (c8*(lsst_r-lsst_i))  +  (c9*(lsst_r-lsst_i)**2)
        ##  sdss_z  = lsst_z + c10 + (c11*(lsst_i-lsst_z)) +  (c12*(lsst_i-lsst_z)**2)


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

    lsst_bands = ["u", "g", "r", "i", "z", "y"]

    # extract the indices of the available lsst bands
    indices = [lsst_bands.index(band) for band in bands]

    lsst = {}
    sdss_responses = []
    for ind in range(len(indices)):
        lsst["{0}".format(lsst_bands[indices[ind]])] = unumpy.uarray(
            lsst_mags[:, ind], lsst_errs[:, ind]
        )
        sdss_responses.append("sdss_%s0" % (lsst_bands[indices[ind]]))

    # transform from lsst to sdss magnitudes
    mgSDSS = (
        lsst["g"]
        + c1
        + (c2 * (lsst["g"] - lsst["r"]))
        + (c3 * (lsst["g"] - lsst["r"]) ** 2)
    )
    mrSDSS = (
        lsst["r"]
        + c4
        + (c5 * (lsst["g"] - lsst["r"]))
        + (c6 * (lsst["g"] - lsst["r"]) ** 2)
    )
    miSDSS = (
        lsst["i"]
        + c7
        + (c8 * (lsst["r"] - lsst["i"]))
        + (c9 * (lsst["r"] - lsst["i"]) ** 2)
    )



    if scaling_relation == "spectroscopic":
        # Find out the K-correction factor using the kcorrect module by Blanton
        # k-correct upto redshift z=0 only
        k_corrections = kcorr_sdss(
            np.array([mgSDSS, mrSDSS, miSDSS]),
            redshift,
            responses=sdss_responses,
            responses_out=sdss_responses,
            band_shift=0.0,
            redshift_range=[0, 2],
        )

        # Apply the K-correction on the SDSS magnitudes
        mgSDSS = mgSDSS - k_corrections[:, 0]
        mrSDSS = mrSDSS - k_corrections[:, 1]
        miSDSS = miSDSS - k_corrections[:, 2]

        ## Note: It will be better if we apply the K-correction directly on the LSST magnitudes,
        ## but no such relation is known to me right now.

        # Use the SDSS g-band and r-band magnitudes to get the B-band apparent magnitude of the galaxy using the relation
        # given in equation A2, Appendix, Bell et al 2004 for red galaxies. This is required only for using the relations
        # based on spectroscopic measurements.

        mag_B = mgSDSS + 0.155 + 0.370 * (mgSDSS - mrSDSS)

        # calculates the distance luminosity using the redshift and the cosmology
        Dlum = (cosmo.luminosity_distance(redshift) * cosmo.H(0) / 100).value

        # Convert the apparent B-band magnitude to the absolute B-band magnitude using the redshift and cosmology defined
        # Note that the 25 here comes since Dlum is in Mpc
        MabsB = mag_B - 5.0 * np.log10(Dlum) - 25.0
        """
        Now using the data from DEEP2 and COMBO-17 surveys, Bell et 2004 found that
        the B-band luminosity function evolves such that characteristic magnitude MBstar
        decline by 1.5 magnitudes from z=0.0 to z=1.0. We use the same assumption here;

        Hence, MBstar and redshift should follow the relation, i.e., MBstar =
        MBstar0-(redshift)*1.5. where MBstar0 = MBstar(at redshift=0). In our case, MBstar0 = -19.31
        has been estimated from the mean value of the MBstar0, from Table 1, Bell et al
        2004.
        """
        # define a 1D line model for MBstar evolution with redshift.
        MBstar_func = Linear1D(-1.5, -19.31)

        # Use the above model to calculate MB* at the deflector redshift
        MBstar = MBstar_func(redshift)

        # Calculate L/L* using the magnitude-luminosity relation
        LbyLstar = 10.0 ** (-0.4 * (MabsB - MBstar))
        """
        Now use the L-sigma relation for the elliptical galaxies i.e., the Faber
        Jackson relation, sigma/sigma_star = (L/Lstar)**(1/alpha) and taking the sigma*
        and alpha value from Choi et al 2007, derived for early type galaxies, calculate
        the the velocity dispersion sigma.
        """
        sigma_star, alpha = ufloat(161, 5), 2.32  # Choi et al 2007

        # Use sigma_star and alpha values to calculate the stellar velocity dispersion sigma
        sigma = sigma_star * LbyLstar ** (1 / alpha)


    elif scaling_relation == "weak-lensing":
        # k-correct upto redshift z=0.1, since the scaling relations used are at z=0.1
        k_corrections = kcorr_sdss(
            np.array([mgSDSS, mrSDSS, miSDSS]),
            redshift,
            responses=sdss_responses,
            responses_out=sdss_responses,
            band_shift=0.1,
            redshift_range=[0, 2],
        )

        # Apply the K-correction on the SDSS magnitudes
        mgSDSS = mgSDSS - k_corrections[:, 0]
        mrSDSS = mrSDSS - k_corrections[:, 1]
        miSDSS = miSDSS - k_corrections[:, 2]

        # calculates the distance luminosity using the redshift and the cosmology
        Dlum = (cosmo.luminosity_distance(redshift) * cosmo.H(0) / 100).value

        # Convert the apparent r-band magnitudes to the absolute r
        Mabsr = mrSDSS - 5.0 * np.log10(Dlum) - 25.0
        """
        We assume the same assumption here (from Bell et al 2004) for decline of
        characteristic magnitude Mrstar for r'-band,

        Hence, Mrstar and redshift should follow the relation, i.e., MBstar =
        MBstar0-(redshift-0.1)*1.5. where Mrstar0 = MBstar(at redshift=0.1).

        In our case, Mrstar0 = -20.44 has been estimated from Table 2, Blanton et al
        2003.
        """

        # Convert the sdss r-mag to r'-mag from Frei & Gunn 2003 (Table 3).
        # r' is a fake filter i.e., r shifted to z=0.1.
        Mabsr = Mabsr - 0.11

        Mrstar0 = -20.44  # calculated at redhift=0.1
        # Use the above value to calculate Mrstar at the deflector redshift
        Mrstar = Mrstar0 - (redshift - 0.1) * 1.5

        # Calculate L/L* using the magnitude-luminosity relation
        LbyLstar = 10.0 ** (-0.4 * (Mabsr - Mrstar))
        """
        Now use the L-sigma relation and taking the sigma_star and alpha value from
        Parker et al 2007, derived using weak-lensing measurements, calculate the the
        velocity dispersion sigma."""
        # sigma_star, alpha = ufloat(142,18), 3      # Parker et al 2007

        sigma_star_nominal = np.ones(len(LbyLstar)) * 142
        sigma_star_stdev = np.ones(len(LbyLstar)) * 18
        alpha = np.ones(len(LbyLstar)) * 3
        sigma_star = unumpy.uarray(sigma_star_nominal, sigma_star_stdev)
        sigma_star[miSDSS > 20.5] = ufloat(137, 11)
        alpha[miSDSS > 20.5] = 3

        # Use sigma_star and alpha values to calculate the stellar velocity dispersion sigma
        sigma = sigma_star * LbyLstar ** (1 / alpha)

    else:
        raise KeyError("Invalid input for scaling relations.")

    # returns the calculated sigma
    # type: a 1D array of uncertainties.core.Variable
    ##   to extract the nomianl values and the uncertainities in separate arrays,
    ##   use unumpy.nominal_values(sigma) and unumpy.std_devs(sigma)
    return sigma
