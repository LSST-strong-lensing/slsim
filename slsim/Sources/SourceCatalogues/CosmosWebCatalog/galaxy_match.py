import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import units as u

from slsim.ImageSimulation.image_quality_lenstronomy import (
    ROMAN_BAND_LIST,
    EUCLID_BAND_LIST,
    LSST_BAND_LIST,
)
from slsim.Util.catalog_util import match_source

# The pixel scale for the detection_images cutouts is 0.03 arcseconds per pixel
PIXEL_SCALE = 0.03

ARCSEC_TO_RADIANS = 4.84814e-6


def process_catalog(cosmo, catalog_path):
    """

    :param cosmo: instance of astropy cosmology
    :param catalog_path: path to the directory containing the COSMOSWeb_galaxy_catalog.fits and corresponding images.
        This directory can be downloaded from https://zenodo.org/records/19188494
    :type catalog_path: string
    :return: astropy table of galaxies.
    """

    catalog_path = os.path.join(catalog_path, "COSMOSWeb_galaxy_catalog.fits")
    catalog = Table.read(catalog_path, format="fits")

    # sersic radius is the radius along the major axis
    # angular size is the geometric mean of the major and minor axes
    catalog["angular_size"] = catalog["sersic_radius"].data * np.sqrt(
        catalog["axis_ratio"].data
    )
    catalog["angular_size"].unit = u.arcsec

    # Convert angular_size to physical size (arcseconds to kPc)
    ang_dist = cosmo.angular_diameter_distance(catalog["z"])
    catalog["physical_size"] = catalog["angular_size"].to(u.rad) * ang_dist.value * 1000
    catalog["physical_size"].unit = u.kiloparsec

    return catalog


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
    :param catalog_path: path to the directory containing the COSMOSWeb_galaxy_catalog.fits and corresponding images.
        This directory can be downloaded from https://zenodo.org/records/19188494
    :param max_scale: The image will be scaled to have the desired angular size. Scaling up
        results in a more pixelated image. This input determines what the maximum up-scale factor is.
    :type max_scale: int or float
    :param match_n_sersic: determines whether to match based off of the sersic index as well.
        Since n_sersic is usually undefined and set to 1 in SLSim, this is set to False by default.
    :type match_n_sersic: bool
    :return: tuple(list, float, float, int) consisting of:
        1. A list of band-specific images matched from the catalog: [F115W, F150W, F277W, F444W]
        2. The scale that the image needs to match the desired angular size
        3. The angle of rotation needed to match the desired angle
        4. The matched astropy row.
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
    id = matched_source["id"]
    image_file = f"COSMOSWeb_galaxy_{id}_image.fits"
    image_file = os.path.join(catalog_path, image_file)

    image_list = []
    with fits.open(image_file) as hdul:

        # The images are stored as [F115W, F150W, F277W, F444W]
        for i in range(4):
            image_list.append(hdul[i + 1].data)

    # Scale the image so that it matches the desired angular size
    # lenstronomy's Interpol class needs the pixel scale, so that gets included here
    scale = PIXEL_SCALE * angular_size / matched_source["angular_size"]

    # Rotate the image so that it matches the desired angle
    # Convert desired angle from slsim convention (North to East) to lenstronomy convention (East to North)
    phi = -matched_source["sersic_angle"] + (np.pi / 2 - sersic_angle)

    return image_list, scale, phi, matched_source


def _select_image_from_band(band, image_list):
    """Selects an image from the image_list based off of the input band.

    This function does not perform color transformations to accurately preserve
    the wavelength profile of the original source. Instead, the images are assigned
    based on the relative wavelengths between the bands of a given detector, with
    the only goal being to have morphology differences between bands.

    :param band: imaging band
    :type band: string
    :param image_list: contains the images for the bands in the following order:
        [F115W, F150W, F277W, F444W].
    type image_list: list of numpy arrays
    :return: image from source catalog corresponding to specific band
    """

    if band in ROMAN_BAND_LIST:
        if band == "F062":
            return image_list[0]
        elif band == "F087":
            return (image_list[0] + image_list[1]) / 2
        elif band == "F106":
            return image_list[1]
        elif band == "F129":
            return (image_list[1] + image_list[2]) / 2
        elif band == "F158":
            return image_list[2]
        elif band == "F184":
            return (image_list[2] + image_list[3]) / 2
        elif band == "F213":
            return image_list[3]
        elif band == "F146":
            return image_list[1] + image_list[2]

    elif band in EUCLID_BAND_LIST:
        if band == "VIS":
            return image_list[0]

    elif band in LSST_BAND_LIST:
        if band == "u":
            return image_list[0]
        elif band == "g":
            return image_list[1]
        elif band == "r":
            return (image_list[1] + image_list[2]) / 2
        elif band == "i":
            return image_list[2]
        elif band == "z":
            return (image_list[2] + image_list[3]) / 2
        elif band == "y":
            return image_list[3]

    else:
        raise ValueError(
            f"Band should be selected from one of the following:\nLSST: {LSST_BAND_LIST}\nRoman: {ROMAN_BAND_LIST}\nEuclid: {EUCLID_BAND_LIST}"
        )
