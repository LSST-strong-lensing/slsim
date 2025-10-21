import os
import numpy as np
from astropy.table import Table, join

from astropy.io import fits

from lenstronomy.Util.param_util import ellipticity2phi_q


def process_cosmos_catalog(cosmo, catalog_path):
    """This function filters out sources in the catalog so that only
    the nearby, well-resolved galaxies with high SNR remain. Thus, we
    perform the following cuts:
    1. redshift < 1
    2. apparent magnitude < 20
    3. half light radius > 10 pixels

    :param cosmo: instance of astropy cosmology
    :param catalog_path: path to the COSMOS_23.5_training_sample directory.
        Example: catalog_path = "/home/data/COSMOS_23.5_training_sample"
    :type catalog_path: string
    :return: merged astropy table with only the well-resolved galaxies
    """

    catalog1_path = os.path.join(catalog_path, "real_galaxy_catalog_23.5.fits")
    catalog2_path = os.path.join(catalog_path, "real_galaxy_catalog_23.5_fits.fits")
    cat1 = Table.read(catalog1_path, format="fits", hdu=1)
    cat2 = Table.read(catalog2_path, format="fits", hdu=1)
    # These sources are excluded because they are too close to other objects
    source_exclusion_list = [
        79,
        309,
        1512,
        5515,
        7138,
        7546,
        9679,
        14180,
        14914,
        19494,
        22095,
        28335,
        32696,
        32778,
        33527,
        34946,
        36344,
        38328,
        40837,
        41255,
        44363,
        44871,
        49652,
        51303,
        52021,
        55803,
        1368,
        1372,
        1626,
        5859,
        6161,
        6986,
        7312,
        8108,
        8405,
        9349,
        9326,
        9349,
        9745,
        9854,
        9948,
        10146,
        10446,
        11209,
        12397,
        14642,
        14909,
        15473,
        17775,
        17904,
        20256,
        20489,
        21597,
        21693,
        22380,
        23054,
        23390,
        23790,
        24110,
        24966,
        26135,
        27222,
        27781,
        28297,
        29550,
        30089,
        30898,
        30920,
        31548,
        32025,
        33699,
        35553,
        36409,
        36268,
        36576,
        37198,
        37969,
        38873,
        40286,
        40286,
        40924,
        41731,
        44045,
        45066,
        45929,
        45929,
        46575,
        47517,
        48137,
        49441,
        52270,
        52730,
        52759,
        52891,
        54924,
        54445,
        55153,
        10584,
        22051,
        22365,
        23951,
        42334,
        42582,
        51492,
        32135,
        37106,
        37593,
        38328,
        45618,
        47829,
        26145,
    ]

    max_z = 1.0
    faintest_apparent_mag = 20
    min_flux_radius = 10.0

    is_ok = np.ones(len(cat2), dtype=bool)
    is_ok &= cat2["zphot"] < max_z
    is_ok &= cat2["mag_auto"] < faintest_apparent_mag
    is_ok &= cat2["flux_radius"] > min_flux_radius

    # Drop any catalog indices that are in the exclusion list
    is_ok &= np.invert(np.isin(np.arange(len(cat2)), source_exclusion_list))

    filtered_catalog = join(cat1[is_ok], cat2[is_ok], keys="IDENT")

    # This is the half light radius that is the geometric mean of the major and minor axis lengths
    # calculated using sqrt(q) * R_half, where R_half is the half-light radius measured along the major axis
    # We then convert this from units of pixels to arcseconds
    q = filtered_catalog["sersicfit"][:, 3]
    R_half = filtered_catalog["sersicfit"][:, 1]
    filtered_catalog["angular_size"] = (
        np.sqrt(q) * R_half * filtered_catalog["PIXEL_SCALE"].data
    )

    # Convert from arcseoncds to kPc
    ang_dist = cosmo.angular_diameter_distance(filtered_catalog["zphot"].data)
    filtered_catalog["physical_size"] = (
        filtered_catalog["angular_size"].data * 4.84814e-6 * ang_dist.value * 1000
    )

    # drop extraneous data
    keep_columns = [
        "IDENT",
        "GAL_FILENAME",
        "GAL_HDU",
        "PIXEL_SCALE",
        "sersicfit",
        "angular_size",
        "physical_size",
    ]

    for col in filtered_catalog.colnames:
        if col not in keep_columns:
            filtered_catalog.remove_column(col)

    return filtered_catalog


def match_cosmos_source(
    angular_size,
    physical_size,
    e1,
    e2,
    n_sersic,
    processed_cosmos_catalog,
    catalog_path,
    max_scale=1,
    match_n_sersic=False,
):
    """This function matches the parameters in source_dict to find a
    corresponding source in the COSMOS catalog. The parameters being
    matched are:

    1. physical size <= size_tol where size_tol starts at 0.5 kPc and increases by 0.2 until match
    2. axis ratio <= q_tol where q_tol starts at 0.1 and increases by 0.05 until match
    3. n_sersic

    When many matches are found, the match with the best n_sersic is taken.

    :param angular_size: angular size of the source [arcsec]
    :param physical_size: physical size of the source [kpc]
    :param e1: eccentricity modulus
    :param e2: eccentricity modulus
    :param source_dict: Source properties. May be a dictionary or an Astropy table.
     This dict or table should contain atleast redshift, a magnitude in any band,
     sersic index, angular size in arcsec, and ellipticities e1 and e2.
     eg: {"z": 0.8, "mag_i": 22, "n_sersic": 1, "angular_size": 0.10,
     "e1": 0.002, "e2": 0.001}. One can provide magnitudes in multiple bands.
    :type source_dict: dict or astropy.table.Table
    :param processed_cosmos_catalog: the returned object from calling process_cosmos_catalog()
    :param catalog_path: path to the COSMOS_23.5_training_sample directory.
     Example: catalog_path = "/home/data/COSMOS_23.5_training_sample"
    :param max_scale: The COSMOS image will be scaled to have the desired angular size. Scaling up
     results in a more pixelated image. This input determines what the maximum up-scale factor is.
    :type max_scale: int or float
    :param match_n_sersic: determines whether to match based off of the sersic index as well.
     Since n_sersic is usually undefined and set to 1 in SLSim, this is set to False by default.
    :type match_n_sersic: bool
    :return: tuple(ndarray, float, float, int)
     This is the raw image matched from the catalog, the scale that the image needs to
     match angular size, the angle of rotation needed to match the desired e1 and e2, and the galaxy ID.
    """

    processed_cosmos_catalog = processed_cosmos_catalog[
        angular_size <= processed_cosmos_catalog["angular_size"].data * max_scale
    ]
    if len(processed_cosmos_catalog) == 0:
        return None, None, None, None

    # Keep sources within the physical size tolerance, all units in kPc
    size_tol = 0.5
    size_difference = np.abs(
        physical_size - processed_cosmos_catalog["physical_size"].data
    )
    matched_catalog = processed_cosmos_catalog[size_difference < size_tol]
    # If no sources, relax the matching condition and try again
    while len(matched_catalog) == 0:
        size_tol += 0.2
        matched_catalog = processed_cosmos_catalog[size_difference < size_tol]

    phi, q = ellipticity2phi_q(e1, e2)
    # Keep sources within the axis ratio tolerance
    q_tol = 0.1
    q_matched_catalog = matched_catalog[
        np.abs(matched_catalog["sersicfit"][:, 3].data - q) <= q_tol
    ]
    # If no sources, relax the tolerance and try again
    while len(q_matched_catalog) == 0:
        q_tol += 0.05
        q_matched_catalog = matched_catalog[
            np.abs(matched_catalog["sersicfit"][:, 3].data - q) <= q_tol
        ]

    if match_n_sersic:
        # Select source based off of best matching n_sersic
        index = np.argsort(np.abs(q_matched_catalog["sersicfit"][:, 2].data - n_sersic))
        matched_source = q_matched_catalog[index][0]
    else:
        # Select source based off of best matching axis ratio
        index = np.argsort(np.abs(q_matched_catalog["sersicfit"][:, 3].data - q))
        matched_source = q_matched_catalog[index][0]

    # load and save image
    fname = matched_source["GAL_FILENAME"]
    hdu = int(matched_source["GAL_HDU"])
    path = os.path.join(catalog_path, fname)
    with fits.open(path) as file:
        image = file[hdu].data  # flux per pixel

    # Scale the angular size of the COSMOS image so that it matches the source_dict
    scale = (
        matched_source["PIXEL_SCALE"] * angular_size / matched_source["angular_size"]
    )

    # Rotate the COSMOS image so that it matches the angle given in source_dict
    phi = np.pi / 2 - matched_source["sersicfit"][7] - phi

    return image, scale, phi, matched_source["IDENT"]


def safe_value(val):
    """This function ensures that a value that we put into a pandas DataFrame
    is safe, i.e doesn't have mismatched datatypes.

    :param val: value to store in df
    :type val: string or float or list or array
    :return: safe value
    """
    if isinstance(val, np.ndarray):
        # Ensure native byte order
        if hasattr(val, "dtype") and val.dtype.byteorder not in ("=", "|"):
            val = val.astype(val.dtype.newbyteorder("="))
        # If array has one element, convert to float
        if val.size == 1:
            return float(val)
    elif isinstance(val, np.generic):
        return float(val)
    return val
