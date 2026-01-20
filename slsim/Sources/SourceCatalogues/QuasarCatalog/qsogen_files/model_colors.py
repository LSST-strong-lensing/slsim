#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on 2020 November 10 Public release 2021 March 13 Edit 2021 November
17 to add DECam and HSC filters Edit 2022 June 08 to also allow filters to be
stored in ./filters/ Edit 2022 July 29 to add GALEX NUV and FUV filters Edit
2025 March 12 to add Euclid VIS filter.

@author: Matthew Temple

# This research has made use of the SVO Filter Profile Service
#  (http://svo2.cab.inta-csic.es/theory/fps/) supported from
#  the Spanish MINECO through grant AyA2014-55216

Any filter of interest needs to be saved in the same directory as e.g.
'SDSS_u.filter'

Default filters are SDSS AB, UKIDSS Vega, WISE Vega ugrizYJHKW12

Conventional to use AB magnitudes for SDSS, VISTA (KiDS), LSST, Euclid
 and Vega magnitudes for UKIDSS and WISE.
Note UKIDSS and non-KiDS VISTA calibrated to 2MASS which assumes Vega has
 zero mag; WISE also assumes Vega has zero mag.

We use the 2007 Vega spectrum to produce Vega zeropoints.

The code is structured to do the following things once, minimising i/o:
    Load filter response functions and associated wav arrays
    Load filter normalisations
    Concatenate and sort wav arrays
    Identify where sorted array corresponds to each filter wav array
And then run the SED model many times, using filter wav arrays as input wavlen:
    looping over z (or any other variable of interest)
    multiply relevant part of each flux array with filter response function
    convert to magnitude, return colours or mags as appropriate
"""

import numpy as np
from scipy.integrate import simps
from qsosed import Quasar_sed

# assume 2007 Vega spectrum has zero magnitude in all bands
Vega_zeropoints = dict(
    GALEX_NUV_Vega=4.638133e-01,
    GALEX_FUV_Vega=9.411333e-02,
    SDSS_u_Vega=7.986921e-04,
    SDSS_g_Vega=1.071783e-02,
    SDSS_r_Vega=8.455695e-03,
    SDSS_i_Vega=4.624980e-03,
    SDSS_z_Vega=6.676051e-04,
    DECam_u_Vega=8.177943e-04,
    DECam_g_Vega=1.903633e-02,
    DECam_r_Vega=1.684878e-02,
    DECam_i_Vega=1.283454e-02,
    DECam_z_Vega=9.031628e-03,
    DECam_Y_Vega=3.316464e-03,
    HSC_g_Vega=1.881597e-02,
    HSC_r_Vega=1.608603e-02,
    HSC_r2_Vega=2.291566e-02,
    HSC_i_Vega=8.728089e-03,
    HSC_i2_Vega=1.396346e-02,
    HSC_z_Vega=3.394019e-03,
    HSC_Y_Vega=2.322250e-03,
    LSST_u_Vega=1.782632e-02,
    LSST_g_Vega=1.423635e-01,
    LSST_r_Vega=9.468552e-02,
    LSST_i_Vega=5.861893e-02,
    LSST_z_Vega=3.521813e-02,
    LSST_y_Vega=1.643208e-02,
    Euclid_VIS_Vega=3.170234e-02,
    Euclid_Y_Vega=7.743718e-03,
    Euclid_J_Vega=7.092825e-03,
    Euclid_H_Vega=4.253855e-03,
    UKIDSS_Z_Vega=1.284425e-03,
    UKIDSS_Y_Vega=1.119803e-03,
    UKIDSS_J_Vega=1.179108e-03,
    UKIDSS_H_Vega=1.409765e-03,
    UKIDSS_K_Vega=6.859715e-04,
    VISTA_Z_Vega=7.195167e-03,
    VISTA_Y_Vega=4.610067e-03,
    VISTA_J_Vega=5.395487e-03,
    VISTA_H_Vega=5.122664e-03,
    VISTA_Ks_Vega=2.727685e-03,
    WISE_W1_Vega=1.810656e-03,
    WISE_W2_Vega=1.156727e-03,
    WISE_W3_Vega=3.901671e-04,
    WISE_W4_Vega=4.489671e-05,
)
AB_zeropoints = dict(
    GALEX_NUV_AB=2.143184e00,
    GALEX_FUV_AB=6.682505e-01,
    SDSS_u_AB=1.857387e-03,
    SDSS_g_AB=9.721695e-03,
    SDSS_r_AB=9.641594e-03,
    SDSS_i_AB=6.418155e-03,
    SDSS_z_AB=1.078642e-03,
    DECam_u_AB=1.104899e-03,
    DECam_g_AB=1.742508e-02,
    DECam_r_AB=1.991803e-02,
    DECam_i_AB=1.868959e-02,
    DECam_z_AB=1.455719e-02,
    DECam_Y_AB=5.599771e-03,
    HSC_g_AB=1.714996e-02,
    HSC_r_AB=1.837621e-02,
    HSC_r2_AB=2.615511e-02,
    HSC_i_AB=1.253355e-02,
    HSC_i2_AB=2.011298e-02,
    HSC_z_AB=5.445445e-03,
    HSC_Y_AB=3.848121e-03,
    LSST_u_AB=3.219424e-02,
    LSST_g_AB=1.298923e-01,
    LSST_r_AB=1.083751e-01,
    LSST_i_AB=8.188370e-02,
    LSST_z_AB=5.619859e-02,
    LSST_y_AB=2.718541e-02,
    Euclid_VIS_AB=4.040447e-02,
    Euclid_Y_AB=1.454023e-02,
    Euclid_J_AB=1.846444e-02,
    Euclid_H_AB=1.634865e-02,
    UKIDSS_Z_AB=2.060514e-03,
    UKIDSS_Y_AB=1.972653e-03,
    UKIDSS_J_AB=2.739415e-03,
    UKIDSS_H_AB=4.897400e-03,
    UKIDSS_K_AB=3.844725e-03,
    VISTA_Z_AB=1.152809e-02,
    VISTA_Y_AB=8.005087e-03,
    VISTA_J_AB=1.255071e-02,
    VISTA_H_AB=1.793209e-02,
    VISTA_Ks_AB=1.460059e-02,
    WISE_W1_AB=2.134738e-02,
    WISE_W2_AB=2.450814e-02,
    WISE_W3_AB=4.859685e-02,
    WISE_W4_AB=2.006357e-02,
)

zeropoints = {**Vega_zeropoints, **AB_zeropoints}


wavarrs, resparrs = dict(), dict()
for band in [
    "GALEX_NUV",
    "GALEX_FUV",
    "SDSS_u",
    "SDSS_g",
    "SDSS_r",
    "SDSS_i",
    "SDSS_z",
    "DECam_u",
    "DECam_g",
    "DECam_r",
    "DECam_i",
    "DECam_z",
    "DECam_Y",
    "HSC_g",
    "HSC_r",
    "HSC_r2",
    "HSC_i",
    "HSC_i2",
    "HSC_z",
    "HSC_Y",
    "LSST_u",
    "LSST_g",
    "LSST_r",
    "LSST_i",
    "LSST_z",
    "LSST_y",
    "Euclid_VIS",
    "Euclid_Y",
    "Euclid_J",
    "Euclid_H",
    "UKIDSS_Z",
    "UKIDSS_Y",
    "UKIDSS_J",
    "UKIDSS_H",
    "UKIDSS_K",
    "VISTA_Z",
    "VISTA_Y",
    "VISTA_J",
    "VISTA_H",
    "VISTA_Ks",
    "WISE_W1",
    "WISE_W2",
    "WISE_W3",
    "WISE_W4",
]:
    try:
        wavarr, response = np.genfromtxt(band + ".filter", unpack=True)
    except OSError:
        wavarr, response = np.genfromtxt("filters/" + band + ".filter", unpack=True)
    wavarrs[band] = wavarr
    resparrs[band] = response


# default zeropoints use AB for SDSS and Vega for UKIDSS and WISE
def get_colours(
    redshifts,
    filters=[
        "SDSS_u_AB",
        "SDSS_g_AB",
        "SDSS_r_AB",
        "SDSS_i_AB",
        "SDSS_z_AB",
        "UKIDSS_Y_Vega",
        "UKIDSS_J_Vega",
        "UKIDSS_H_Vega",
        "UKIDSS_K_Vega",
        "WISE_W1_Vega",
        "WISE_W2_Vega",
    ],
    **kwargs
):
    """Get synthetic colours from quasar model.

    Parameters
    ----------
    redshifts : iterable
        List or array of redshifts
    filters : array, optional
        List of filter passbands to compute colours between.
    **kwargs
        Arguments to pass to Quasar_sed.

    Returns
    -------
    model_colours : ndarray
        Array with model colours for each redshift in redshifts.

    Notes
    -----
    Wavelength array in Quasar_sed must cover rest-frame 4000-5000 Angstroms,
    if gflag is set to True.
    get_colours concatenates and sorts the wavelength arrays of the filter
    response functions, and uses this as the input wavelength array to
    Quasar_sed. This is computationally much faster as it means the model is
    only evaluated at the wavelengths it's actually needed and avoids
    unnecessary interpolation. However, this can lead to errors if the host
    galaxy is turned on and an unusually sparse combination of filters is
    requested.
    """

    waves, responses = [], []
    for band in filters:
        band = band.replace("_AB", "")
        band = band.replace("_Vega", "")
        waves.append(wavarrs[band])
        responses.append(resparrs[band])

    obs_wavlen = np.concatenate(waves)
    isort = obs_wavlen.argsort().argsort()
    # indices to invert the sorting on the concatenated array
    split_indices = np.cumsum([len(wav) for wav in waves[:-1]])
    # indices to invert the concatenation

    model_colours = []

    try:
        for z in redshifts:
            rest_ordered_wav = np.sort(obs_wavlen / (1 + z))
            # now in rest frame of qso
            ordered_flux = Quasar_sed(wavlen=rest_ordered_wav, z=z, **kwargs).flux
            # qsosed will produce redshifted flux

            # Create individual arrays with the flux in each observed passband
            fluxes = np.split(ordered_flux[isort], split_indices)

            model_colours.append(-np.diff(sed2mags(filters, waves, fluxes, responses)))
    except TypeError:
        z = float(redshifts)
        rest_ordered_wav = np.sort(obs_wavlen / (1 + z))
        # now in rest frame of qso
        ordered_flux = Quasar_sed(wavlen=rest_ordered_wav, z=z, **kwargs).flux
        # qsosed will produce redshifted flux

        # Create individual arrays with the flux in each observed passband
        fluxes = np.split(ordered_flux[isort], split_indices)

        model_colours.append(-np.diff(sed2mags(filters, waves, fluxes, responses)))

    return np.array(model_colours)


def get_mags(
    redshifts,
    filters=[
        "SDSS_u_AB",
        "SDSS_g_AB",
        "SDSS_r_AB",
        "SDSS_i_AB",
        "SDSS_z_AB",
        "UKIDSS_Y_Vega",
        "UKIDSS_J_Vega",
        "UKIDSS_H_Vega",
        "UKIDSS_K_Vega",
        "WISE_W1_Vega",
        "WISE_W2_Vega",
    ],
    **kwargs
):
    """Get synthetic magnitudes from quasar model.

    Parameters
    ----------
    redshifts : iterable
        List or array of redshifts
    filters : array, optional
        List of filter passbands to compute colours between.
    **kwargs
        Arguments to pass to Quasar_sed.

    Returns
    -------
    model_mags : ndarray
        Array with model magnitudes for each redshift in redshifts.

    Notes
    -----
    Wavelength array in Quasar_sed must cover rest-frame 4000-5000 Angstroms,
    if gflag is set to True.
    get_mags concatenates and sorts the wavelength arrays of the filter
    response functions, and uses this as the input wavelength array to
    Quasar_sed. This is computationally much faster as it means the model is
    only evaluated at the wavelengths it's actually needed and avoids
    unnecessary interpolation. However, this can lead to errors if the host
    galaxy is turned on and an unusually sparse combination of filters is
    requested.
    """

    waves, responses = [], []
    for band in filters:
        band = band.replace("_AB", "")
        band = band.replace("_Vega", "")
        waves.append(wavarrs[band])
        responses.append(resparrs[band])

    obs_wavlen = np.concatenate(waves)
    isort = obs_wavlen.argsort().argsort()
    # indices to invert the sorting on the concatenated array
    split_indices = np.cumsum([len(wav) for wav in waves[:-1]])
    # indices to invert the concatenation

    model_mags = []

    try:
        for z in redshifts:
            rest_ordered_wav = np.sort(obs_wavlen / (1 + z))
            # now in rest frame of qso
            # note wavlength array must cover 4000-5000 Angstroms
            ordered_flux = Quasar_sed(wavlen=rest_ordered_wav, z=z, **kwargs).flux
            # qsosed will produce redshifted flux

            # Create individual arrays with the flux in each observed passband
            fluxes = np.split(ordered_flux[isort], split_indices)

            model_mags.append(sed2mags(filters, waves, fluxes, responses))

    except TypeError:
        z = float(redshifts)
        rest_ordered_wav = np.sort(obs_wavlen / (1 + z))
        # now in rest frame of qso
        # note wavlength array must cover 4000-5000 Angstroms
        ordered_flux = Quasar_sed(wavlen=rest_ordered_wav, z=z, **kwargs).flux
        # qsosed will produce redshifted flux

        # Create individual arrays with the flux in each observed passband
        fluxes = np.split(ordered_flux[isort], split_indices)

        model_mags.append(sed2mags(filters, waves, fluxes, responses))

    return np.array(model_mags)


def sed2mags(filters, waves, fluxes, responses):

    mags = np.full(len(waves), np.nan)

    for i in range(len(waves)):
        flux = simps(waves[i] * responses[i] * fluxes[i], waves[i])
        mags[i] = -2.5 * np.log10(flux / zeropoints[filters[i]])

    return mags


def produce_zeropoints(
    system="Vega",
    filters=[
        "GALEX_NUV",
        "GALEX_FUV",
        "SDSS_u",
        "SDSS_g",
        "SDSS_r",
        "SDSS_i",
        "SDSS_z",
        "DECam_u",
        "DECam_g",
        "DECam_r",
        "DECam_i",
        "DECam_z",
        "DECam_Y",
        "HSC_g",
        "HSC_r",
        "HSC_r2",
        "HSC_i",
        "HSC_i2",
        "HSC_z",
        "HSC_Y",
        "LSST_u",
        "LSST_g",
        "LSST_r",
        "LSST_i",
        "LSST_z",
        "LSST_y",
        "Euclid_VIS",
        "Euclid_Y",
        "Euclid_J",
        "Euclid_H",
        "UKIDSS_Z",
        "UKIDSS_Y",
        "UKIDSS_J",
        "UKIDSS_H",
        "UKIDSS_K",
        "VISTA_Z",
        "VISTA_Y",
        "VISTA_J",
        "VISTA_H",
        "VISTA_Ks",
        "WISE_W1",
        "WISE_W2",
        "WISE_W3",
        "WISE_W4",
    ],
):
    """Produce the Vega and AB zero points for Sloan, UKIDSS and WISE.

    Zero points are pre-computed to save time. If you want to compute
    model photometry in additional filters, first use this function to
    compute their zeropoints and add to dictionaries above.
    """

    waves, responses = [], []
    for band in filters:
        waves.append(wavarrs[band])
        responses.append(resparrs[band])
    print(system + "_zeropoints = dict(")

    if system == "Vega":
        wav_Vega, flux_Vega = np.genfromtxt("vega_2007.lis", unpack=True)
        # Vega spectrum
        fluxes = [np.interp(wav, wav_Vega, flux_Vega) for wav in waves]

        for i in range(len(filters)):
            F = simps(waves[i] * responses[i] * fluxes[i], waves[i])
            print("    " + filters[i] + "_Vega={:.6e},".format(F))

    elif system == "AB":
        const = 0.1088544752  # 3631Jy in erg/s/cm2/A
        # AB system has constant f_nu, so convert to f_lambda
        for i in range(len(filters)):
            F = const * simps(waves[i] ** (-1) * responses[i], waves[i])
            print("    " + filters[i] + "_AB={:.6e},".format(F))
    else:
        raise Exception('System must be "Vega" or "AB"')
    print(")")


if __name__ == "__main__":
    produce_zeropoints("Vega")
    produce_zeropoints("AB")
    print("Derived delta_m = m_AB - m_Vega conversions are as follows, assuming")
    print("Vega has zero magnitude in all bands (consistent with Hewett+ 2006)")
    for band in AB_zeropoints:
        band = band.replace("_AB", "")
        print(
            band, -2.5 * np.log10(zeropoints[band + "_Vega"] / zeropoints[band + "_AB"])
        )

"""
Derived delta_m = m_AB - m_Vega conversions are as follows, assuming
Vega has zero magnitude in all bands (consistent with Hewett+ 2006)
GALEX_NUV 1.6617906508900804
GALEX_FUV 2.1282203802545143
SDSS_u 0.916307531970178
SDSS_g -0.10591218115027329
SDSS_r 0.1424988257715102
SDSS_i 0.35575085540603535
SDSS_z 0.5208942026802013
DECam_u 0.32669625325466756
DECam_g -0.09602111037243544
DECam_r 0.18169480639014704
DECam_i 0.40804866596053224
DECam_z 0.5182787728252919
DECam_Y 0.5687374491977623
HSC_g -0.10065875128954174
HSC_r 0.14451767734892634
HSC_r2 0.14356045219765853
HSC_i 0.39288732951464894
HSC_i2 0.3962084411908027
HSC_z 0.5132977668531478
HSC_Y 0.5483493698742289
LSST_u 0.6417911975387529
LSST_g -0.09953812415364369
LSST_r 0.14661485653583178
LSST_i 0.36289892923935885
LSST_z 0.5073978180142668
LSST_y 0.5466083665474974
Euclid_VIS 0.2633452374412946
Euclid_Y 0.6840543681829497
Euclid_J 1.0387922398669232
Euclid_H 1.461748035493578
UKIDSS_Z 0.5131670489550898
UKIDSS_Y 0.6147726766033728
UKIDSS_J 0.9152606087248839
UKIDSS_H 1.3520471317250795
UKIDSS_K 1.8713980229183953
VISTA_Z 0.5117911999718673
VISTA_Y 0.5991470475407443
VISTA_J 0.9165941107322068
VISTA_H 1.3603375970963005
VISTA_Ks 1.8214404755932185
WISE_W1 2.6787715820675913
WISE_W2 3.3151986991103968
WISE_W3 5.238393684944
WISE_W4 6.625484235337513
"""
