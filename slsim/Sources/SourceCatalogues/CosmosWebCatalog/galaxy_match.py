import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import Cutout2D

from lenstronomy.Util.param_util import ellipticity2phi_q
from slsim.Util.catalog_util import match_source

# The pixel scale for the detection_images cutouts is 0.03 arcseconds per pixel
PIXEL_SCALE = 0.03


def process_catalog(cosmo, catalog_path):
    """This function filters out sources in the catalog so that only
    the nearby, well-resolved galaxies with high SNR remain. Thus, we
    perform the following cuts:

    1. redshift < 1
    2. apparent magnitude < 20 in at least one band

    NOTE: The images in the detection_images directory are coadds of the four NIRCam bands F115W, F150W, F277W, F444W.
    See section 3.3 of https://arxiv.org/pdf/2506.03243. The pixel scale of these images is 0.03 arcseconds.

    :param cosmo: instance of astropy cosmology
    :param catalog_path: path to the directory containing the COSMOSWeb_mastercatalog_v1.fits and detection_images.
        This catalog can be downloaded from:
        https://cosmos2025:780kgalaxies!@cosmos2025.iap.fr/catalog_download.html
        Documentation for this catalog can be found at:
        https://cosmos2025.iap.fr/catalog.html
    :type catalog_path: string
    :return: merged astropy table with only the well-resolved galaxies.
        This astropy table is the same as the one that comes with the catalog,
        but with some columns renamed for clarity, and extraneous columns removed.
    """

    catalog_path = os.path.join(catalog_path, "COSMOSWeb_mastercatalog_v1.fits")
    photometry_table = Table.read(catalog_path, format="fits", hdu=1)
    lephare_table = Table.read(catalog_path, format="fits", hdu=2)

    max_z = 1.0
    faintest_apparent_mag = 20

    is_ok = np.ones(len(lephare_table), dtype=bool)

    # filters out sources with artifacts/issues
    # see https://cosmos2025.iap.fr/catalog.html#quality-flags for warn_flag documentation
    is_ok &= photometry_table["warn_flag"].data == 0

    # redshift cut
    is_ok &= lephare_table["zfinal"].data < max_z

    # Some galaxies have their redshifts in the catalog set to -99
    # The documentation does not explain why; filter these galaxies out
    is_ok &= lephare_table["zfinal"].data != -99

    # only includes galaxies
    # type = 0 is galaxies, type = 1 is stars, type = 2 is QSOs
    is_ok &= lephare_table["type"].data == 0

    # magnitude cuts: apparent magnitude < 20 in at least one band
    is_ok2 = np.zeros_like(is_ok)
    for band in [
        "mag_auto_f115w",
        "mag_auto_f150w",
        "mag_auto_f277w",
        "mag_auto_f444w",
        "mag_auto_f770w",
        "mag_auto_hst-f814w",
    ]:
        is_ok2 |= photometry_table[band].data < faintest_apparent_mag
    is_ok &= is_ok2

    # Drop any remaining catalog sources that have nans within a 100x100 cutout
    # also drop sources that have other nearby sources/contaminants within 75 pixels
    source_exclusion_list_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "source_exclusion_list.npy"
    )
    with open(source_exclusion_list_file, "rb") as f:
        source_exclusion_list = np.load(f)
    is_ok &= np.invert(np.isin(np.arange(len(photometry_table)), source_exclusion_list))

    photometry_table = photometry_table[is_ok]
    lephare_table = lephare_table[is_ok]

    # NOTE: catalog has different ellipticity and angle conventions; we overwrite them here
    e1 = -np.array(photometry_table["e2"], dtype=np.float64)
    e2 = np.array(photometry_table["e1"], dtype=np.float64)
    phi, q = ellipticity2phi_q(e1=e1, e2=e2)
    photometry_table["sersic_angle"] = phi
    photometry_table["axis_ratio"] = q

    # radius_sersic is the half-light radius along the major axis
    # convert it to the geometric mean of the major and minor axis lengths
    # also convert from degrees to arcseconds and rename
    photometry_table["radius_sersic"] = (
        np.sqrt(q) * photometry_table["radius_sersic"].data * 3600
    )
    photometry_table.rename_column("radius_sersic", "angular_size")

    # Convert angular_size to physical size (arcseconds to kPc)
    ang_dist = cosmo.angular_diameter_distance(lephare_table["zfinal"].data)
    photometry_table["physical_size"] = (
        photometry_table["angular_size"].data * 4.84814e-6 * ang_dist.value * 1000
    )

    photometry_table.rename_column("fmf_chi2", "sersic_fit_chi2")
    photometry_table.rename_column("sersic", "sersic_index")

    # drop extraneous data
    keep_columns = [
        "id",
        "tile",
        "x_image",  # coordinates of the source on the detection image in units of pixels
        "y_image",  # coordinates of the source on the detection image in units of pixels
        "sersic_index",  # sersic index n
        "axis_ratio",  # axis ratio q
        "sersic_angle",  # radians, measured clockwise from north with origin at bottom left
        "angular_size",  # half light radius (geometric mean) in arcseconds
        "physical_size",  # kpc
        "sersic_fit_chi2",  # reduced chi^2 of the sersic model
    ]

    for col in photometry_table.colnames:
        if col not in keep_columns:
            photometry_table.remove_column(col)

    return photometry_table


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
    corresponding source in the COSMOSWeb catalog. The parameters being
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
    :param catalog_path: path to the directory containing the COSMOSWeb_mastercatalog_v1.fits and detection_images.
        This catalog can be downloaded from:
        https://cosmos2025:780kgalaxies!@cosmos2025.iap.fr/catalog_download.html
        Documentation for this catalog can be found at:
        https://cosmos2025.iap.fr/catalog.html
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
    tile = matched_source["tile"]
    image_file = catalog_path + f"/detection_images/detection_chi2pos_SWLW_{tile}.fits"
    data = fits.getdata(image_file)

    # Create cutout centered at coords
    size = (100, 100)  # size in pixels (height, width)
    coords = (matched_source["x_image"], matched_source["y_image"])
    image = Cutout2D(data, coords, size).data

    # Scale the angular size of the COSMOS image so that it matches the source_dict
    # lenstronomy's Interpol class needs the pixel scale, so that gets included here
    scale = PIXEL_SCALE * angular_size / matched_source["angular_size"]

    # Rotate the COSMOS image so that it matches the angle given in source_dict
    phi = matched_source["sersic_angle"] - sersic_angle

    return image, scale, phi, matched_source["id"]
