import os
import numpy as np
from astropy.table import Table, join
from astropy.io import fits

from slsim.Util.catalog_util import match_source


def process_catalog(cosmo, catalog_path):
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
    :return: merged astropy table with only the well-resolved galaxies.
        This astropy table is the same as the one that comes with the catalog,
        but with some columns renamed for clarity, and extraneous columns removed.
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

    # angular_size is the half light radius that is the geometric mean of the major and minor axis lengths
    # calculated using sqrt(q) * R_half, where R_half is the half-light radius measured along the major axis
    # We then convert this from units of pixels to arcseconds
    q = filtered_catalog["sersicfit"][:, 3]
    R_half = filtered_catalog["sersicfit"][:, 1]
    filtered_catalog["angular_size"] = (
        np.sqrt(q) * R_half * filtered_catalog["PIXEL_SCALE"].data
    )

    # Convert angular_size to physical size (arcseconds to kPc)
    ang_dist = cosmo.angular_diameter_distance(filtered_catalog["zphot"].data)
    filtered_catalog["physical_size"] = (
        filtered_catalog["angular_size"].data * 4.84814e-6 * ang_dist.value * 1000
    )

    # Rename columns
    filtered_catalog["axis_ratio"] = filtered_catalog["sersicfit"][:, 3]
    filtered_catalog["sersic_index"] = filtered_catalog["sersicfit"][:, 2]
    filtered_catalog["sersic_angle"] = filtered_catalog["sersicfit"][:, 7]

    # drop extraneous data
    keep_columns = [
        "IDENT",
        "GAL_FILENAME",
        "GAL_HDU",
        "PIXEL_SCALE",
        "axis_ratio",
        "sersic_index",
        "sersic_angle",
        "angular_size",  # half light radius (geometric mean) in arcseconds
        "physical_size",  # kpc
    ]

    for col in filtered_catalog.colnames:
        if col not in keep_columns:
            filtered_catalog.remove_column(col)

    return filtered_catalog


def load_source(
    angular_size,
    physical_size,
    axis_ratio,
    sersic_angle,
    n_sersic,
    processed_catalog,
    catalog_path,
    max_scale=1,
    match_n_sersic=False,
):
    """This function matches the parameters in source_dict to find a
    corresponding source in the COSMOS catalog. The parameters being
    matched are:

    1. physical size
    2. axis ratio
    3. n_sersic only if match_n_sersic is True

    Each parameter being matched is normalized so that the max is 1 and min is 0.
    Matching is then performed by selecting the nearest point in 2D space (3D if match_n_sersic is True).

    :param angular_size: desired angular size of the source [arcsec]
    :param physical_size: desired physical size of the source [kpc]
    :param axis_ratio: desired axis ratio
    :param sersic_angle: desired sersic angle
    :param processed_catalog: the returned object from calling process_catalog()
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

    matched_source = match_source(
        angular_size,
        physical_size,
        axis_ratio,
        n_sersic,
        processed_catalog,
        max_scale,
        match_n_sersic,
    )
    if matched_source is None:
        return None, None, None, None

    # load and save image
    fname = matched_source["GAL_FILENAME"]
    hdu = int(matched_source["GAL_HDU"])
    path = os.path.join(catalog_path, fname)
    with fits.open(path) as file:
        image = file[hdu].data  # flux per pixel

    # Scale the angular size of the COSMOS image so that it matches the source_dict
    # lenstronomy's Interpol class needs the pixel scale, so that gets included here
    scale = (
        matched_source["PIXEL_SCALE"] * angular_size / matched_source["angular_size"]
    )

    # Rotate the COSMOS image so that it matches the angle given in source_dict
    phi = np.pi / 2 - matched_source["sersic_angle"] - sersic_angle

    return image, scale, phi, matched_source["IDENT"]
