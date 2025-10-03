#!/usr/bin/env python
import numpy as np
from slsim.Deflectors.MassLightConnection.light2mass import get_velocity_dispersion
from uncertainties import unumpy
from astropy.cosmology import FlatLambdaCDM
from slsim.Util.color_transformations import LSST_to_SDSS
from slsim.Util.k_correction import kcorr_sdss


def get_galaxy_parameters_from_moments(xx, xy, yy):
    """Calculate the parameters for a galaxy using the HSM moments.

    : param xx : xx component of the Hirata-Seljak-Mandelbaum (HSM) moments in the reference band
    : type xx :  float

    : param xy : xy component of the HSM moments in the reference band
    : type xy : float

    : param yy : yy component of the HSM moments in the reference band
    : type yy : float

    returns: tuple
            (position angle, effective radius, axis ratio, and the ellipticity)
    """

    # Construct the 2x2 second-moment matrix Q
    Q = np.array([[xx, xy], [xy, yy]])

    # Compute eigenvalues and eigenvectors of Q
    eig_val, eig_vec = np.linalg.eigh(Q)

    # Major and minor axes
    a = np.sqrt(np.max(eig_val))  # Major axis
    b = np.sqrt(np.min(eig_val))  # Minor axis
    a_indx = np.argmax(eig_val)

    # Calculate cosine and sine of the position angle
    # The angle is measured counterclockwise from the positive x-axis.
    cos_t, sin_t = eig_vec[0, a_indx], -eig_vec[1, a_indx]

    # Compute ellipticity
    ellipticity = 1 - np.sqrt(eig_val[0] / eig_val[1])

    # Compute position angle (theta) in degrees, effective radius (r_eff) in pixels, and axial ratio (q)
    theta = np.rad2deg(np.arctan2(sin_t, cos_t))  # Position angle
    r_eff = np.sqrt(a * b)  # Effective radius
    q = b / a  # Axial ratio

    # Return the parameters
    return theta, r_eff, q, ellipticity


def compute_magnitude(flux, flux_err, zeropoint=31.4):
    """Convert LSST flux to magnitude and compute errors.

    params:

    :param flux : from DP0.2 catalogs
    type flux: float or array-like

    :param flux_err : float or array-like
    type flux_err : flux error

    :param zeropoint : zeropoint of the AB magnitude system
    :type zeropoint : float

    Returns:
    -------
    tuple (magnitude, magnitude_error)
        - magnitude : float or array-like
            The computed magnitude using the LSST zero-point calibration.
        - magnitude_error : float or array-like
            The propagated uncertainty in the magnitude.
    """
    magnitude = -2.50 * np.log10(flux) + zeropoint
    mag_error = (2.5 / np.log(10)) * (flux_err / flux)
    return magnitude, mag_error


def find_massive_ellipticals(
    DP0_table,
    fracDev_limit=0.8,
    iMag_cut_massive=-22,
    pixel_scale=0.2,
    cosmo=FlatLambdaCDM(H0=72, Om0=0.26),
    parallel=False,
):
    """This function identifies potential lensing galaxies (deflectors) from
    the LSST DP0.2 object catalogs or any other catalog, based on the
    morphology, and provides their lens model parameters using the light-to-
    mass function in SLSim.

    : param DP0_table: the foreground galaxy catalog extracted from DP0.2 Object catalog
         having the g, r, i, z, y bands cModel, deVaucouleur, and exponential fluxes and errors,
         the true/ or photometric redshift,
         redshift, gmag, rmag, imag, zmag, ymag, err-g, err-r, err-i, err-z, err-y
         Please check the query in 'extract_catalogs_DP0.py' to extract the required parameters
         from DP0.2 Object Catalogs
         NOTE: the input catalog should have column names, as returned from DP0.2 Object catalogs'
    : type DP0_table: astropy table

    : param  fracDev_limit: the fracDev limit above which a galaxy is considered to be an elliptical galaxy
            (default is 0.8)
    : type fracDev_limit: float

    : param magnitude_cut_massive: absolute magnitude cut for selecting massive (luminous) galaxies
    : type magnitude_cut_massive: float

    : param pixel_scale: pixel scale for the LSST
    : type pixel_scale: float

    : param cosmo: the cosmology model used
    : type cosmo: astropy.cosmology

    : param parallel: will be set to True only if you are running the code in parallel
         default is False

    Returns: An output catalog (astropy table) of all the **massive ellipticals** selected from the input catalog
            having the same columns as in the input catalog, with additional columns added for
            velocity dispersion, position angle, axis ratio, half light radius (in arcsec), ellipticity.
    : type : astropy table
    """

    id_valid = np.where(
        (~np.isnan(DP0_table["obj_u_cModelFlux"]))
        & (~np.isnan(DP0_table["obj_u_cModelFluxerr"]))
        & (~np.isnan(DP0_table["obj_g_cModelFlux"]))
        & (~np.isnan(DP0_table["obj_g_cModelFluxerr"]))
        & (~np.isnan(DP0_table["obj_r_cModelFlux"]))
        & (~np.isnan(DP0_table["obj_r_cModelFluxerr"]))
        & (~np.isnan(DP0_table["obj_i_cModelFlux"]))
        & (~np.isnan(DP0_table["obj_i_cModelFluxerr"]))
        & (~np.isnan(DP0_table["obj_z_cModelFlux"]))
        & (~np.isnan(DP0_table["obj_z_cModelFluxerr"]))
        & (~np.isnan(DP0_table["obj_y_cModelFlux"]))
        & (~np.isnan(DP0_table["obj_y_cModelFluxerr"]))
        & (DP0_table["obj_u_cModelFlux"] > DP0_table["obj_u_cModelFluxerr"])
        & (DP0_table["obj_g_cModelFlux"] > DP0_table["obj_g_cModelFluxerr"])
        & (DP0_table["obj_r_cModelFlux"] > DP0_table["obj_r_cModelFluxerr"])
        & (DP0_table["obj_i_cModelFlux"] > DP0_table["obj_i_cModelFluxerr"])
        & (DP0_table["obj_z_cModelFlux"] > DP0_table["obj_z_cModelFluxerr"])
        & (DP0_table["obj_y_cModelFlux"] > DP0_table["obj_y_cModelFluxerr"])
    )[0]
    DP0_table = DP0_table[id_valid]

    # Compute fracDev for each galaxy to classify ellipticals
    # Example band mapping
    band_mapping = {
        "u": "obj_u",
        "g": "obj_g",
        "r": "obj_r",
        "i": "obj_i",
        "z": "obj_z",
        "y": "obj_y",
    }

    # Initialize a list to store fractional de Vaucouleurs profile contribution (fracDev) values
    fracDev_list = []

    for row in DP0_table:
        band_prefix = band_mapping.get(row["obj_refband"])

        fracDev = (row[f"{band_prefix}_cModelFlux"] - row[f"{band_prefix}_bdFluxD"]) / (
            row[f"{band_prefix}_bdFluxB"] - row[f"{band_prefix}_bdFluxD"]
        )
        fracDev_list.append(fracDev)

    # Select ellipticals based on fracDev threshold
    id_ellipticals = np.where(
        (np.array(fracDev_list) > fracDev_limit) & (np.array(fracDev_list) <= 1.2)
    )[0]
    DP0_ellipticals = DP0_table[id_ellipticals]

    bands = ["u", "g", "r", "i", "z"]
    magnitudes, mag_errors = {}, {}

    for band in bands:
        magnitudes[band], mag_errors[band] = compute_magnitude(
            DP0_ellipticals[f"obj_{band}_cModelFlux"].value,
            DP0_ellipticals[f"obj_{band}_cModelFluxerr"].value,
        )

    # Step 4: Convert LSST magnitudes to SDSS magnitudes
    sdss_mags = LSST_to_SDSS(
        unumpy.uarray(magnitudes["u"], mag_errors["u"]),
        unumpy.uarray(magnitudes["g"], mag_errors["g"]),
        unumpy.uarray(magnitudes["r"], mag_errors["r"]),
        unumpy.uarray(magnitudes["i"], mag_errors["i"]),
        unumpy.uarray(magnitudes["z"], mag_errors["z"]),
    )
    # Use the redshift to calculate the luminosity distance
    redshift = DP0_ellipticals["ts_redshift"].value
    luminosity_distance = cosmo.luminosity_distance(redshift).to("pc").value

    # Find the k_correction coefficients for the five SDSS bands for the ellipticals
    k_correction_coefficients = kcorr_sdss(sdss_mags, redshift, band_shift=0.0)

    # Calculate the r-band absolute magnitude of the ellipticals
    # the luminosity_distance is in 'kpc';
    Mag_i_sdss = (
        sdss_mags[3]
        - k_correction_coefficients[:, 3]
        - 5.0 * np.log10(luminosity_distance / 10)
    )

    # set a cut on absolute magnitude to choose highly luminous/ massive ellipticals
    # this cut is chosen based on stellar mass vs i-band absolute magnitude for ellipticals
    # galaxies in the cosmo DC2 catalog
    id_massive = np.where(Mag_i_sdss <= iMag_cut_massive)[0]
    DP0_massive_ellipticals = DP0_ellipticals[id_massive]

    magnitude_array = np.array([magnitudes[band][id_massive] for band in bands])
    magnitude_error_array = np.array([mag_errors[band][id_massive] for band in bands])
    redshift_array = DP0_massive_ellipticals["ts_redshift"]
    redshift_array = np.array(redshift_array)

    # use the SLSim light-to-mass model to get the velocity dispersion of the massive ellipticals
    massive_ellipticals_velocity_dispersion = get_velocity_dispersion(
        deflector_type="elliptical",
        lsst_mags=magnitude_array,
        lsst_errs=magnitude_error_array,
        redshift=redshift_array,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        bands=["u", "g", "r", "i", "z"],
        scaling_relation="spectroscopic",
    )

    DP0_massive_ellipticals["velocity dispersion"] = (
        massive_ellipticals_velocity_dispersion
    )

    # Get indices where all shape parameters are finite (not NaN)
    id_valid_shape = np.where(
        np.isfinite(DP0_massive_ellipticals["obj_shape_xx"])
        & np.isfinite(DP0_massive_ellipticals["obj_shape_xy"])
        & np.isfinite(DP0_massive_ellipticals["obj_shape_yy"])
    )[0]

    DP0_massive_ellipticals = DP0_massive_ellipticals[id_valid_shape]

    # Compute their position angle, effective radius, axis ratio, ellipticity
    theta_all, r_eff_all, q_all, ellipticity_all = [], [], [], []
    for i in range(len(DP0_massive_ellipticals)):
        xx, xy, yy = (
            DP0_massive_ellipticals["obj_shape_xx"][i],
            DP0_massive_ellipticals["obj_shape_xy"][i],
            DP0_massive_ellipticals["obj_shape_yy"][i],
        )

        theta, r_eff, q, ellipticity = get_galaxy_parameters_from_moments(xx, xy, yy)
        theta_all.append(theta)
        r_eff_all.append(r_eff)  # in pixels
        q_all.append(q)
        ellipticity_all.append(ellipticity)

    # convert the effective radius from pixels to arcsec
    r_eff_all = np.array(r_eff_all) * pixel_scale

    # bring the position angle to convention 'East of north'
    theta_all = (np.array(theta_all) + 180) % 360
    theta_all = theta_all % 180

    DP0_massive_ellipticals["position angle"] = theta_all
    DP0_massive_ellipticals["effective radius"] = r_eff_all
    DP0_massive_ellipticals["axis ratio"] = q_all
    DP0_massive_ellipticals["ellipticity"] = ellipticity_all

    return DP0_massive_ellipticals
