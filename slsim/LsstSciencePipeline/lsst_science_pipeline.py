import numpy as np
from astropy.table import Table, vstack
from slsim.ImageSimulation.image_simulation import (
    sharp_image,
    lens_image,
)
from slsim.Util.param_util import transformmatrix_to_pixelscale, degrade_coadd_data
from scipy import interpolate
from scipy.stats import norm, halfnorm
import matplotlib.pyplot as plt
from slsim.ImageSimulation.image_simulation import point_source_coordinate_properties
from slsim.Util.param_util import (
    random_ra_dec,
    fits_append_table,
    detect_object,
)
import h5py
import os

try:
    import lsst.geom as geom
    from lsst.pipe.tasks.insertFakes import _add_fake_sources
    from lsst.afw.math import Warper
    import galsim
except ModuleNotFoundError:
    lsst = None
    galsim = None
import pyvo

"""
This module provides necessary functions to inject lenses to the dp0 data. For this, it 
uses some of the packages provided by the LSST Science Pipeline.
"""

RSP_TAP_SERVICE = "https://data.lsst.cloud/api/tap"
homedir = os.path.expanduser("~")
token_file = os.path.join(homedir, ".rsp-tap.token")
with open(token_file, "r") as f:
    token_str = f.readline()
cred = pyvo.auth.CredentialStore()
cred.set_password("x-oauth-basic", token_str)
credential = cred.get("ivo://ivoa.net/sso#BasicAA")
rsp_tap = pyvo.dal.TAPService(RSP_TAP_SERVICE, session=credential)


def DC2_cutout(ra, dec, num_pix, butler, band):
    """Draws a cutout from the DC2 data based on the given ra, dec pair.

    For this, one needs to provide a butler to this function. To
    initiate Butler, you need to specify data configuration and
    collection of the data.
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param: band: image band
    :returns: cutout image
    """
    skymap = butler.get("skyMap")
    point = geom.SpherePoint(ra, dec, geom.degrees)
    cutout_size = geom.ExtentI(num_pix, num_pix)
    # print(cutout_size)

    # Read this from the table we have at hand...
    tract_Info = skymap.findTract(point)
    patch_Info = tract_Info.findPatch(point)
    my_tract = tract_Info.tract_id
    my_patch = patch_Info.getSequentialIndex()
    xy = geom.PointI(tract_Info.getWcs().skyToPixel(point))
    bbox = geom.BoxI(xy - cutout_size // 2, cutout_size)
    coadd_Id_r = {"tract": my_tract, "patch": my_patch, "band": band}
    coadd_cut_r = butler.get("deepCoadd", parameters={"bbox": bbox}, dataId=coadd_Id_r)
    return coadd_cut_r


def lens_inejection(
    lens_pop, num_pix, delta_pix, butler, ra, dec, lens_cut=None, flux=None
):
    """Chooses a random lens from the lens population and injects it to a DC2
    cutout image. For this one needs to provide a butler to this function. To
    initiate Butler, you need to specify data configuration and collection of
    the data.

    :param lens_pop: lens population from slsim
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param lens_cut: list of criteria for lens selection
    :param flux: flux need to be asigned to the lens image. It sould be
        None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lens in r-band, DC2
        cutout image in r-band, cutout image with injected lens in r, g
        , and i band
    """
    # lens = sim_lens
    if lens_cut is None:
        kwargs_lens_cut = {}
    else:
        kwargs_lens_cut = lens_cut

    rgb_band_list = ["r", "g", "i"]
    lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
    skymap = butler.get("skyMap")
    point = geom.SpherePoint(ra, dec, geom.degrees)
    cutoutSize = geom.ExtentI(num_pix, num_pix)
    # Read this from the table we have at hand
    tractInfo = skymap.findTract(point)
    patchInfo = tractInfo.findPatch(point)
    my_tract = tractInfo.tract_id
    my_patch = patchInfo.getSequentialIndex()
    xy = geom.PointI(tractInfo.getWcs().skyToPixel(point))
    bbox = geom.BoxI(xy - cutoutSize // 2, cutoutSize)
    injected_final_image = []
    # band_report = []
    box_center = []
    cutout_image = []
    lens_image = []
    for band in rgb_band_list:
        coaddId_r = {"tract": my_tract, "patch": my_patch, "band": band}

        # coadd cutout image
        coadd_cut_r = butler.get(
            "deepCoadd", parameters={"bbox": bbox}, dataId=coaddId_r
        )
        lens = sharp_image(
            lens_class=lens_class,
            band=band,
            mag_zero_point=27,
            delta_pix=delta_pix,
            num_pix=num_pix,
        )
        if flux is None:
            gsobj = galsim.InterpolatedImage(
                galsim.Image(lens), scale=delta_pix, normalization="flux"
            )
        else:
            gsobj = galsim.InterpolatedImage(
                galsim.Image(lens), scale=delta_pix, flux=flux
            )

        wcs_r = coadd_cut_r.getWcs()
        bbox_r = coadd_cut_r.getBBox()
        x_min_r = bbox_r.getMinX()
        y_min_r = bbox_r.getMinY()
        x_max_r = bbox_r.getMaxX()
        y_max_r = bbox_r.getMaxY()

        # Calculate the center coordinates
        x_center_r = (x_min_r + x_max_r) / 2
        y_center_r = (y_min_r + y_max_r) / 2

        center_r = geom.Point2D(x_center_r, y_center_r)
        # geom.Point2D(26679, 15614)
        point_r = wcs_r.pixelToSky(center_r)
        ra_degrees = point_r.getRa().asDegrees()
        dec_degrees = point_r.getDec().asDegrees()
        center = (ra_degrees, dec_degrees)

        # image_r = butler.get("deepCoadd", parameters={'bbox':bbox_r},dataId=coaddId_r)
        arr_r = np.copy(coadd_cut_r.image.array)

        _add_fake_sources(coadd_cut_r, [(point_r, gsobj)])
        inj_arr_r = coadd_cut_r.image.array
        injected_final_image.append(inj_arr_r)
        # band_report.append(band)
        box_center.append(center)
        cutout_image.append(arr_r)
        lens_image.append((inj_arr_r - arr_r))

    t = Table(
        [
            [lens_image[0]],
            [cutout_image[0]],
            [injected_final_image[0]],
            [injected_final_image[1]],
            [injected_final_image[2]],
            [box_center[0]],
        ],
        names=(
            "lens",
            "cutout_image",
            "injected_lens_r",
            "injected_lens_g",
            "injected_lens_i",
            "cutout_center",
        ),
    )
    return t


def get_dp0_images(butler, ra, dec, band_list, coadd_injection):
    """Retrieve coadd or visit images for the given bands.

    :param butler: butler object
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param band_list: List of imaging band in which lens need to be
        injected.
    :param coadd_injection: Boolean. If True, queries the DP0 coadd
        image and if False, queries the single visit DP0 images.
    :return: coadd or single visit image, number of images in coadd,
        magnitude zero point, and variance map in specified bands.
    """
    skymap = butler.get("skyMap")
    point = geom.SpherePoint(ra, dec, geom.degrees)
    # cutoutSize = geom.ExtentI(num_pix, num_pix)

    tractInfo = skymap.findTract(point)
    patchInfo = tractInfo.findPatch(point)
    my_tract = tractInfo.tract_id
    my_patch = patchInfo.getSequentialIndex()
    coadd, coadd_nImage, mag_zero_visit, variance_map = [], [], [], []

    for band in band_list:
        coaddId = {"tract": my_tract, "patch": my_patch, "band": band}
        coadd_image = butler.get("deepCoadd", dataId=coaddId)

        if not coadd_injection:
            visit_info = coadd_image.getInfo().getCoaddInputs().ccds
            index = np.random.randint(0, len(visit_info))
            dataId_visit = {
                "visit": visit_info[index]["visit"],
                "detector": visit_info[index]["ccd"],
                "band": visit_info[index]["filter"],
                "tract": tractInfo.getId(),
            }
            visit_image = butler.get("calexp", dataId=dataId_visit)
            coadd.append(visit_image)
            variance_map.append(visit_image.getVariance())
            mag_zero_visit.append(
                2.5 * np.log10(visit_image.getPhotoCalib().getInstFluxAtZeroMagnitude())
            )
        else:
            coadd.append(coadd_image)
            variance_map.append(coadd_image.getVariance())
            coadd_nImage.append(butler.get("deepCoadd_nImage", dataId=coaddId))

    return coadd, coadd_nImage, mag_zero_visit, variance_map


def generate_cutout_bbox(x_center, y_center, num_pix):
    """Generate bounding box for cutout selection.

    :param x_center: x value of center of the cutout box in pixel unit
    :param y_center: y value of center of the cutout box in pixel unit
    :param num_pix: number of pixel for the cutout
    :return: 2d box readable by the butler
    """
    xbox_min = x_center - ((num_pix - 1) / 2)
    xbox_max = x_center + ((num_pix - 1) / 2)
    ybox_min = y_center - ((num_pix - 1) / 2)
    ybox_max = y_center + ((num_pix - 1) / 2)

    return geom.Box2I(
        geom.Point2I(xbox_min, ybox_min), geom.Point2I(xbox_max, ybox_max)
    )


def lens_inejection_fast(
    lens_pop,
    num_pix,
    mag_zero_point,
    transform_pix2angle,
    butler,
    ra,
    dec,
    num_cutout_per_patch=10,
    lens_cut=None,
    noise=True,
    coadd_injection=True,
    coadd_year=5,
    band_list=["r", "g", "i"],
    center_box_size=3,
    center_source_snr_threshold=5,
    false_positive=False,
):
    """Chooses a random lens from the lens population and injects it to a DC2
    cutout image. For this one needs to provide a butler to this function. To
    initiate Butler, you need to specify data configuration and collection of
    the data.

    :param lens_pop: lens population from slsim. It can be a LensPop
        instance or list of Lens class.
    :param num_pix: number of pixel for the cutout
    :param mag_zero_point: magnitude zero point in band
    :param transform_pix2angle: transformation matrix (2x2) of pixels
        into coordinate displacements
    :param butler: butler object
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param num_cutout_per_patch: number of cutout image drawn per patch
    :param lens_cut: list of criteria for lens selection
    :param noise: poisson noise to be added to an image. If True,
        poisson noise will be added to the image based on exposure time.
    :param coadd_injection: Boolean. If True, injects lens into the DP0
        coadd image and if False, injects lens into the single visit DP0
        images.
    :param coadd_year: Year for the coadd images. This parameter is used
        to rescale the noise properties of 5 year dp0 coadd images to
        desired year of coadd.
    :param band_list: List of imaging band in which lens need to be
        injected.
    :param center_box_size: Size of the central box in arcsec (default
        is 3 arcsec).
    :param center_source_snr_threshold: SNR threshold for object
        detection in center box (default is 5).
    :param false_positive: Boolean. If false, code assumes that the provided
     population is a lens population. If True, code assumes that the provided
     population is a false popitive poulation. False positive contains an
     elliptical galaxy at the center and blue galaxies around this central galaxy.
     for more detail, please see: slsim/FalsePositives/
    :returns: An astropy table containing Injected lens in r-band, DC2
        cutout image in r-band, cutout image with injected lens in r, g
        , and i band
    """

    pixel_scale = transformmatrix_to_pixelscale(transform_pix2angle)
    if lens_cut is None:
        kwargs_lens_cut = {}
    else:
        kwargs_lens_cut = lens_cut

    rgb_band_list = band_list
    coadd, coadd_nImage, mag_zero_visit, variance_map = get_dp0_images(
        butler=butler,
        ra=ra,
        dec=dec,
        band_list=rgb_band_list,
        coadd_injection=coadd_injection,
    )
    bbox = coadd[0].getBBox()
    xmin, ymin = bbox.getBegin()
    xmax, ymax = bbox.getEnd()
    wcs = coadd[0].getWcs()

    valid_cutouts = 0
    table = []
    # for i in range(len(x_center)):
    while valid_cutouts < num_cutout_per_patch:
        # Randomly select a position for the cutout
        x_center = np.random.randint(xmin + 150, xmax - 150)
        y_center = np.random.randint(ymin + 150, ymax - 150)
        cutout_bbox = generate_cutout_bbox(
            x_center=x_center, y_center=y_center, num_pix=num_pix
        )
        if isinstance(lens_pop, list):
            lens_class = lens_pop[valid_cutouts]
        else:
            if false_positive is False:
                lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
            else:
                lens_class = lens_pop.draw_false_positive()
        injected_final_image, box_center, cutout_image_list, lens_image, lens_id = (
            [],
            [],
            [],
            [],
            [],
        )
        is_valid = True
        for j, band in enumerate(band_list):
            cutout_image = coadd[j][cutout_bbox]
            cutout_variance = variance_map[j][cutout_bbox]
            # Check for existing objects in the cutout image
            if detect_object(
                cutout_image.image.array,
                cutout_variance.array,
                pixel_scale=pixel_scale,
                box_size_arcsec=center_box_size,
                snr_threshold=center_source_snr_threshold,
            ):
                is_valid = False
                break  # Discard this cutout and try again
            if noise is True:
                if coadd_injection is True:
                    exposure_map = 30 * coadd_nImage[j][cutout_bbox].array
                    zero_point_magnitude = mag_zero_point
                else:
                    exposure_map = 30
                    zero_point_magnitude = mag_zero_visit[j]
            else:
                exposure_map = None
            final_injected_image = add_object(
                cutout_image,
                lens_class=lens_class,
                band=rgb_band_list[j],
                mag_zero_point=zero_point_magnitude,
                num_pix=num_pix,
                transform_pix2angle=transform_pix2angle,
                exposure_time=exposure_map,
                coadd_year=coadd_year,
            )
            center_point = geom.Point2D(x_center, y_center)
            center_wcs = wcs.pixelToSky(center_point)
            ra_deg = center_wcs.getRa().asDegrees()
            dec_deg = center_wcs.getDec().asDegrees()

            injected_final_image.append(final_injected_image)
            box_center.append((ra_deg, dec_deg))
            cutout_image_list.append(cutout_image.image.array)
            lens_image.append((final_injected_image - cutout_image.image.array))
            lens_id.append(lens_class.generate_id())
        if is_valid:
            # Define column names dynamically based on band_list
            prefix = "injected_object" if false_positive else "injected_lens"
            column_names = (
                ["lens_id", "lens", "cutout_image"]
                + [f"{prefix}_{band}" for band in band_list]
                + ["cutout_center"]
            )

            # Construct row data dynamically
            data = (
                [[[lens_id[0]]], [lens_image[0]], [cutout_image_list[0]]]
                + [[img] for img in injected_final_image]
                + [[box_center[0]]]
            )

            # Create Table instance
            table_1 = Table(data, names=column_names)
            table.append(table_1)
            valid_cutouts += 1  # Increase count of successful cutouts
    lens_catalog = vstack(table)
    return lens_catalog


def lens_injection_fast_coadd(
    lens_pop,
    num_pix,
    coadd,
    coadd_nImage,
    mag_zero_visit,
    variance_map,
    transform_pix2angle,
    num_cutout_per_patch=5,
    lens_cut=None,
    noise=True,
    coadd_year=5,
    band_list=["r", "g", "i"],
    center_box_size=3,
    center_source_snr_threshold=5,
    false_positive=False,
):
    """Injects lenses into randomly taken cutouts from a given DC2 cutout
    image.

    :param lens_pop: lens population from slsim. It can be a LensPop
        instance or list of Lens class.
    :param num_pix: number of pixel for the cutout
    :param coadd: list of coadd images in different bands
    :param coadd_nImage: list of number of images in coadd in different
        bands
    :param mag_zero_visit: list of magnitude zero points in different
        bands
    :param variance_map: list of variance maps in different bands
    :param transform_pix2angle: transformation matrix (2x2) of pixels
        into coordinate displacements
    :param num_cutout_per_patch: number of cutout image drawn per patch
    :param lens_cut: list of criteria for lens selection
    :param noise: poisson noise to be added to an image. If True,
        poisson noise will be added to the image based on exposure time.
    :param coadd_year: Year for the coadd images. This parameter is used
        to rescale the noise properties of 5 year dp0 coadd images to
        desired year of coadd.
    :param band_list: List of imaging band in which lens need to be
        injected.
    :param center_box_size: Size of the central box in arcsec (default
        is 3 arcsec).
    :param center_source_snr_threshold: SNR threshold for object
        detection in center box (default is 5).
    :param false_positive: Boolean. If false, code assumes that the
        provided population is a lens population. If True, code assumes
        that the provided population is a false positive population.
        False positive contains an elliptical galaxy at the center and
        blue galaxies around this central galaxy. for more detail,
        please see: slsim/FalsePositives/
    :returns: An astropy table containing Injected lens in r-band, DC2
        cutout image in r-band, cutout image with injected lens in r, g
        , and i band, cutout box center and cutout bbox.
    """

    pixel_scale = transformmatrix_to_pixelscale(transform_pix2angle)
    if lens_cut is None:
        kwargs_lens_cut = {}
    else:
        kwargs_lens_cut = lens_cut

    rgb_band_list = band_list

    bbox = coadd[0].getBBox()
    xmin, ymin = bbox.getBegin()
    xmax, ymax = bbox.getEnd()
    wcs = coadd[0].getWcs()

    cutout_dim = (num_pix + 1) // 2
    valid_cutouts = 0
    table = []
    # for i in range(len(x_center)):
    while valid_cutouts < num_cutout_per_patch:
        # Randomly select a position for the cutout
        x_center = np.random.randint(xmin + cutout_dim, xmax - cutout_dim)
        y_center = np.random.randint(ymin + cutout_dim, ymax - cutout_dim)
        cutout_bbox = generate_cutout_bbox(
            x_center=x_center, y_center=y_center, num_pix=num_pix
        )
        if isinstance(lens_pop, list):
            lens_class = lens_pop[valid_cutouts]
        else:
            if false_positive is False:
                lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
            else:
                lens_class = lens_pop.draw_false_positive()
        (
            injected_final_image,
            box_center,
            cutout_image_list,
            lens_image,
            lens_,
            psf_kernel_,
            noise_map_,
        ) = ([], [], [], [], [], [], [])
        is_valid = True
        for j, band in enumerate(band_list):
            cutout_image = coadd[j][cutout_bbox]
            cutout_variance = variance_map[j][cutout_bbox]
            # Check for existing objects in the cutout image
            if detect_object(
                cutout_image.image.array,
                cutout_variance.array,
                pixel_scale=pixel_scale,
                box_size_arcsec=center_box_size,
                snr_threshold=center_source_snr_threshold,
            ):
                is_valid = False
                break  # Discard this cutout and try again

            if noise is True:

                exposure_map = 30 * coadd_nImage[j][cutout_bbox].array
                zero_point_magnitude = mag_zero_visit[j]

            else:
                exposure_map = None

            final_injected_image, psf_kernel, noise_map = add_object(
                cutout_image,
                lens_class=lens_class,
                band=rgb_band_list[j],
                mag_zero_point=zero_point_magnitude,
                num_pix=num_pix,
                transform_pix2angle=transform_pix2angle,
                exposure_time=exposure_map,
                coadd_year=coadd_year,
            )
            center_point = geom.Point2D(x_center, y_center)
            center_wcs = wcs.pixelToSky(center_point)
            ra_deg = center_wcs.getRa().asDegrees()
            dec_deg = center_wcs.getDec().asDegrees()

            injected_final_image.append(final_injected_image)
            box_center.append((ra_deg, dec_deg))
            cutout_image_list.append(cutout_image.image.array)
            lens_image.append((final_injected_image - cutout_image.image.array))
            lens_.append(lens_class)
            psf_kernel_.append(psf_kernel)
            noise_map_.append(noise_map)

        if is_valid:
            # Define column names dynamically based on band_list
            prefix = "injected_object" if false_positive else "injected_lens"
            column_names = (
                ["lens_class", "lens", "cutout_image"]
                + [f"{prefix}_{band}" for band in band_list]
                + [f"psf_kernel_{band}" for band in band_list]
                + [f"noise_map_{band}" for band in band_list]
                + ["cutout_center", "cutout_bbox"]
            )

            # Construct row data dynamically
            data = (
                [[lens_[0]], [lens_image[0]], [cutout_image_list[0]]]
                + [[img] for img in injected_final_image]
                + [[img] for img in psf_kernel_]
                + [[img] for img in noise_map_]
                + [[box_center[0]], [cutout_bbox]]
            )

            # Create Table instance
            table_1 = Table(data, names=column_names)
            table.append(table_1)
            valid_cutouts += 1  # Increase count of successful cutouts
    lens_catalog = vstack(table)

    return lens_catalog


def multiple_lens_injection(
    lens_pop, num_pix, delta_pix, butler, ra, dec, lens_cut=None, flux=None
):
    """Injects random lenses from the lens population to multiple DC2 cutout
    images using lens_inejection function. For this one needs to provide a
    butler to this function. To initiate Butler, you need to specify data
    configuration and collection of the data.

    :param lens_pop: lens population from slsim
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param ra: ra for a cutout
    :param dec: dec for a cutout
    :param flux: flux need to be asigned to the lens image. It sould be
        None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lenses in r-band, DC2
        cutout images in r-band, cutout images with injected lens in r,
        g , and i band for a given set of ra and dec
    """
    injected_images = []
    for i in range(len(ra)):
        injected_images.append(
            lens_inejection(
                lens_pop,
                num_pix,
                delta_pix,
                butler,
                ra[i],
                dec[i],
                lens_cut=lens_cut,
                flux=flux,
            )
        )
    injected_image_catalog = vstack(injected_images)
    return injected_image_catalog


def multiple_lens_injection_fast(
    lens_pop,
    num_pix,
    mag_zero_point,
    transform_pix2angle,
    butler,
    ra,
    dec,
    num_cutout_per_patch=10,
    lens_cut=None,
    noise=True,
    coadd_injection=True,
    coadd_year=5,
    band_list=["r", "g", "i"],
    center_box_size=3,
    center_source_snr_threshold=5,
    output_file=None,
    false_positive=False,
):
    """Injects random lenses from the lens population to multiple DC2 cutout
    images using lens_inejection_fast function. For this one needs to provide a
    butler to this function. To initiate Butler, you need to specify data
    configuration and collection of the data.

    :param lens_pop: lens population from slsim. It can be a LensPop
        instance or list of Lens class.
    :param num_pix: number of pixel for the cutout
    :param mag_zero_point: magnitude zero point in band
    :param transform_pix2angle: transformation matrix (2x2) of pixels
        into coordinate displacements
    :param butler: butler object
    :param ra: ra for a cutout
    :param dec: dec for a cutout
    :param noise: poisson noise to be added to an image. If True,
        poisson noise will be added to the image based on exposure time.
    :param coadd_injection: Boolean. If True, injects lens into the DP0
        coadd image and if False, injects lens into the single visit DP0
        images.
    :param coadd_year: Year for the coadd images. This parameter is used
        to rescale the noise properties of 5 year dp0 coadd images to
        desired year of coadd.
    :param band_list: List of imaging band in which lens need to be
        injected.
    :param center_box_size: Size of the central box in arcsec (default
        is 3 arcsec).
    :param center_source_snr_threshold: SNR threshold for object
        detection in center box (default is 5).
    :param output_file: path to the output FITS file where data will be
        saved
    :param false_positive: Boolean. If false, code assumes that the provided
     population is a lens population. If True, code assumes that the provided
     population is a false popitive poulation.
    :returns: An astropy table containing Injected lenses in r-band, DC2
        cutout images in r-band, cutout images with injected lens in r,
        g , and i band for a given set of ra and dec. If output_file
        path is provided, it saves this astropy table in fits file with
        the given name.
    """
    injected_images = []
    for i in range(len(ra)):
        if isinstance(lens_pop, list):
            lens_pop_array = np.array(lens_pop)
            lens_pop_group = lens_pop_array.reshape(len(ra), num_cutout_per_patch)
            _lens_pop = list(lens_pop_group[i])
        else:
            _lens_pop = lens_pop
        injected_image = lens_inejection_fast(
            _lens_pop,
            num_pix,
            mag_zero_point,
            transform_pix2angle,
            butler,
            ra[i],
            dec[i],
            num_cutout_per_patch,
            lens_cut=lens_cut,
            noise=noise,
            coadd_injection=coadd_injection,
            coadd_year=coadd_year,
            band_list=band_list,
            center_box_size=center_box_size,
            center_source_snr_threshold=center_source_snr_threshold,
            false_positive=false_positive,
        )
        if output_file is None:
            injected_images.append(injected_image)
        else:
            first_table = not os.path.exists(output_file)
            if first_table:
                injected_image.write(output_file, overwrite=True)
                first_table = False
            else:
                fits_append_table(output_file, injected_image)
    if len(injected_images) > 1:
        injected_image_catalog = vstack(injected_images)
        return injected_image_catalog
    return None


def add_object(
    image_object,
    lens_class,
    band,
    mag_zero_point,
    num_pix,
    transform_pix2angle,
    exposure_time,
    calibFluxRadius=None,
    image_type="dp0",
    coadd_year=5,
):
    """Injects a given object in a dp0 cutout image or SLSimObject.

    :param image_object: cutout image from the dp0 data or SLSimObject.
     eg: slsim_object = SLSimObject(image_array, psfkernel, pixelscale).
    :param lens_class: Lens() object
    :param band: imaging band
    :param mag_zero_point: list of magnitude zero point for sqeuence of exposure
    :param num_pix: number of pixels per axis
    :param transform_pix2angle: list of transformation matrix (2x2) of pixels into
        coordinate displacements for each exposure
    :param exposure_time: list of exposure time for each exposure. It could be single
        exposure time or a exposure map.
    :param variance_map: variance map for a given image_object.
    :param calibFluxRadius: (optional) Aperture radius (in pixels) used to define the
        calibration for thisexposure+catalog. This is used to produce the correct
        instrumental fluxes within the radius. The value should match that of the field
        defined in slot_CalibFlux_instFlux.
    :param image_type: dp0 or slsim_object.
    :param coadd_year: Year for the coadd images. This parameter is used to rescale the
     noise properties of 5 year dp0 coadd images to desired year of coadd.
    :returns: an image with injected source
    """
    if image_type == "dp0":
        wcs = image_object.getWcs()
        psf = image_object.getPsf()
        bbox = image_object.getBBox()
        xmin, ymin = bbox.getBegin()
        xmax, ymax = bbox.getEnd()
        x_cen, y_cen = (xmin + xmax) / 2, (ymin + ymax) / 2
        pt = geom.Point2D(x_cen, y_cen)
        psfArr = psf.computeKernelImage(pt).array
        if calibFluxRadius is not None:
            apCorr = psf.computeApertureFlux(calibFluxRadius, pt)
            psf_ker = psfArr / apCorr
        else:
            psf_ker = psfArr
        pixscale = wcs.getPixelScale(bbox.getCenter()).asArcseconds()
    elif image_type == "slsim_object":
        psf_ker = image_object.psf_kernel
        pixscale = image_object.pixel_scale
    else:
        raise ValueError(
            "Provided image object is not supported. Either use dp0 image"
            "object or SLSimObject"
        )
    num_pix_cutout = np.shape(image_object.image.array)[0]
    delta_pix = transformmatrix_to_pixelscale(transform_pix2angle)
    lens_im = lens_image(
        lens_class=lens_class,
        band=band,
        mag_zero_point=mag_zero_point,
        num_pix=num_pix,
        psf_kernel=psf_ker,
        transform_pix2angle=transform_pix2angle,
        exposure_time=exposure_time,
    )
    objects = [(lens_im, delta_pix)]
    for lens, pix_scale in objects:
        num_pix_lens = np.shape(lens)[0]
        if num_pix_cutout != num_pix_lens:
            raise ValueError(
                "Images with different pixel number cannot be combined. Please make"
                "sure that your lens and dp0 cutout image have the same pixel number."
                f"lens pixel number = {num_pix_lens} and dp0 image pixel number ="
                f"{num_pix_cutout}"
            )
        if abs(pixscale - pix_scale) >= 10**-4:
            raise ValueError(
                "Images with different pixel scale should not be combined. Please make"
                "sure that your lens image and dp0 cutout image have compatible pixel"
                "scale."
            )
        else:
            variance_map = image_object.getVariance()
            noise_map = np.sqrt(variance_map.array)
            degraded_image = degrade_coadd_data(
                image_object.image.array,
                variance_map=variance_map.array,
                exposure_map=exposure_time,
                original_num_years=5,
                degraded_num_years=coadd_year,
                use_noise_diff=True,
            )
            injected_image = degraded_image[0] + lens
            return injected_image, psf_ker, noise_map


def cutout_image_psf_kernel(
    dp0_image,
    lens_class,
    band,
    mag_zero_point,
    delta_pix,
    num_pix,
    transform_pix2angle,
    calibFluxRadius=12,
):
    """This function extracts psf kernels from the dp0 cutout image at point
    source image positions and deflector position. dp0 images are objects that
    has various attributes. In the dp0.2 data, psf kernel vary with coordinate
    and can be computed using given psf model.

    :param dp0_image: cutout image from the dp0 data.
    :param lens_class: class object containing all information of the
        lensing system (e.g., Lens())
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :param calibFluxRadius: (optional) Aperture radius (in pixels) used
        to define the calibration for thisexposure+catalog. This is used
        to produce the correct instrumental fluxes within the radius.
        The value should match that of the field defined in
        slot_CalibFlux_instFlux.
    :returns: Astropy table containing psf kernel at image and deflector
        positions.
    """
    image_data = point_source_coordinate_properties(
        lens_class=lens_class,
        band=band,
        mag_zero_point=mag_zero_point,
        delta_pix=delta_pix,
        num_pix=num_pix,
        transform_pix2angle=transform_pix2angle,
    )
    # get the property of cutout image
    bbox = dp0_image.getBBox()
    # wcs = dp0_image.getWcs()
    xmin_cut, ymin_cut = bbox.getBegin()
    xmax_cut, ymax_cut = bbox.getEnd()
    dp0_image_psf = dp0_image.getPsf()
    grid_shape = np.shape(dp0_image.image.array)
    x_center = (xmin_cut + xmax_cut) / 2
    y_center = (ymin_cut + ymax_cut) / 2

    # map pixel grid to dp0 pixel coordinate
    x_original = np.arange(xmin_cut, xmax_cut + 1)
    x_rescale = np.arange(0, grid_shape[1] + 1)
    f_x = interpolate.interp1d(x_rescale, x_original)

    y_original = np.arange(ymin_cut, ymax_cut + 1)
    y_rescale = np.arange(0, grid_shape[0] + 1)
    f_y = interpolate.interp1d(y_rescale, y_original)

    ## transform image pix coordinate of point source image to dp0 pixel coodinate.
    image_list = image_data["image_pix"]
    image_rescaled_to_dp0_cord = []
    for i in range(len(image_list)):
        image_rescaled_to_dp0_cord.append(
            (float(f_x(image_list[i][0])), float(f_y(image_list[i][1])))
        )

    psf_kernels = []
    for i in range(len(image_rescaled_to_dp0_cord)):
        psf_kernels.append(
            dp0_image_psf.computeKernelImage(
                geom.Point2D(
                    image_rescaled_to_dp0_cord[i][0], image_rescaled_to_dp0_cord[i][1]
                )
            ).array
        )

    pt = geom.Point2D(x_center, y_center)
    psf_kernel_for_deflector = dp0_image_psf.computeKernelImage(pt).array
    ap_Corr = dp0_image_psf.computeApertureFlux(calibFluxRadius, pt)
    psf_kernel_for_deflector /= ap_Corr
    table_of_kernels = Table(
        [[psf_kernels], [psf_kernel_for_deflector]],
        names=("psf_kernel_for_images", "psf_kernel_for_deflector"),
    )
    return table_of_kernels


def tap_query(center_coords, radius=0.1, band="i"):
    """This function uses tap_service from RSP to query calexp visit
    information around a coordinate point.

    :param center_coords: A coordinate point around which visit
        informations are needed.
    :type center_coords: string. eg: "65, -36"
    :param radius: Radius around center point for query. The unit of
        radius is degree.
    :param band: imaging band
    :return: An astropy table of visit information sorted with
        observation time.
    """
    service = rsp_tap
    query = (
        "SELECT ra, decl,"
        + "ccdVisitId, band, "
        + "visitId, physical_filter, detector, "
        + "expMidptMJD, expTime, zeroPoint, skyRotation "
        + "FROM dp02_dc2_catalogs.CcdVisit "
        + "WHERE CONTAINS(POINT('ICRS', ra, decl), "
        + "CIRCLE('ICRS', "
        + center_coords
        + ", "
        + radius
        + ")) = 1"
        + "AND band = "
        + "'"
        + str(band)
        + "' "
    )
    result = service.search(query)
    result_table = result.to_table()
    sorted_column = np.argsort(result_table["expMidptMJD"])
    expo_information = result_table[sorted_column]
    return expo_information


def list_of_calexp(expo_information, butler):
    """Extracts calexp images based on exposure information.

    :param expo_information: Astropy table containing exposure
        information. It must contain Visit ID and Detector ID.
    :param butler: butler object
    :return: list of calexp images.
    """
    calexp_image = []
    for visitid, detectorid in zip(
        expo_information["visitId"], expo_information["detector"]
    ):
        dataId = {"visit": visitid, "detector": detectorid}
        calexp_image.append(butler.get("calexp", dataId=dataId))
    return calexp_image


def warp_to_exposure(exposure, warp_to_exposure):
    """This function aligns two given dp0 images.

    :param exposure: image that need to be aligned
    :param warp_to_exposure: Reference image on which exposure should be
        aligned.
    :return: Image aligned to reference image.
    """
    warper = Warper(warpingKernelName="lanczos4")
    warped_exposure = warper.warpExposure(
        warp_to_exposure.getWcs(), exposure, destBBox=warp_to_exposure.getBBox()
    )
    return warped_exposure


def aligned_calexp(calexp_image):
    """Alignes list of given images to the first image in the list.

    :param calexp_image: list of calexp images.
    :return: list of aligned images.
    """
    selected_calexp = calexp_image[1:]
    aligned_calexp_image = [calexp_image[0]]
    for i in range(0, len(selected_calexp), 1):
        aligned_calexp_image.append(
            warp_to_exposure(selected_calexp[i], calexp_image[0])
        )
    return aligned_calexp_image


def dp0_center_radec(calexp_image):
    """Computes the center ra, dec of given dp0 image.

    :param calexp_image: dp0 image
    :return: A sphere point of center ra, dec of given image
    """
    bbox = calexp_image.getBBox()
    xmin, ymin = bbox.getBegin()
    xmax, ymax = bbox.getEnd()
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    radec_point = geom.SpherePoint(calexp_image.getWcs().pixelToSky(x_center, y_center))
    return radec_point


def calexp_cutout(calexp_image, radec, size):
    """Creates the same size cutouts from given list of dp0 images.

    :param calexp_image: list of dp0 images
    :param radec: SpherePoint of radec around which we want a cutout
    :param size: cutout size in pixel unit
    :return: cutout image of a given size
    """
    cutout_extent = geom.ExtentI(size, size)
    cutout_calexp_image = []
    for i in range(len(calexp_image)):
        cutout_calexp_image.append(calexp_image[i].getCutout(radec, cutout_extent))
    return cutout_calexp_image


def radec_to_pix(radec, dp0_image):
    """Converts ra, dec to pixel units for a fiven image or list of images.

    :param radec: SpherePoint of radec
    :type radec: an object: eg: geom.SpherePoint(65*degree, -36*degree)
    :param dp0_image: an image or list of images containing given radec
    :return: corresponding Point2D of pixel coordinate in provided
        images. If an image is provided, output will be a single
        Point2D. If list of image is provided, output will be list of
        correspoding Point2D.
    """
    if isinstance(dp0_image, list):
        pix = []
        for images in dp0_image:
            pix.append(images.getWcs().skyToPixel(radec))
    else:
        pix = dp0_image.getWcs().skyToPixel(radec)
    return pix


def dp0_psf_kernels(pixel_coord, dp0_image):
    """Extracts psf kernels of given dp0 image at given pixel coordinate.

    :param pixel_coord: Pixel coordinate. eg:Point2D(2129.7674135086986,
        2506.6640697199136)
    :param dp0_image: dp0 image to extract psf
    :return: list of psf kernels for given images
    """
    psf_kernels = []
    for points, images in zip(pixel_coord, dp0_image):
        dp0_image_psf = images.getPsf()
        psf_kernels.append(dp0_image_psf.computeKernelImage(points).array)
    return psf_kernels


def dp0_time_series_images_data(butler, center_coord, radius="0.1", band="i", size=101):
    """Creates time series cutouts and associated metadata from dp0 data.

    :param butler: butler object
    :param center_coord: A coordinate point around which we need to
        create time series images.
    :param radius: radius for query
    :param band: imaging band
    :param size: cutout size of images
    :return: An astropy table containg time series images and other
        information
    """
    expo_information = tap_query(center_coords=center_coord, radius=radius, band=band)
    calexp_image = list_of_calexp(expo_information, butler=butler)
    radec = dp0_center_radec(calexp_image[0])
    radec_list = radec_list = [(radec.getRa().asDegrees(), radec.getDec().asDegrees())]
    radec_list.extend(radec_list * (len(calexp_image) - 1))
    cutout_image = calexp_cutout(calexp_image, radec, 450)
    aligned_image = aligned_calexp(cutout_image)
    aligned_image_cutout = calexp_cutout(aligned_image, radec, size)
    pixels = radec_to_pix(radec, calexp_image)
    psf_kernel = dp0_psf_kernels(pixels, calexp_image)
    obs_time = expo_information["expMidptMJD"]
    expo_time = expo_information["expTime"]
    zero_point_mag = expo_information["zeroPoint"]
    dp0_time_series_cutout = []
    for i in range(len(aligned_image_cutout)):
        dp0_time_series_cutout.append(aligned_image_cutout[i].image.array)
    table_data = Table(
        [
            dp0_time_series_cutout,
            psf_kernel,
            obs_time,
            expo_time,
            zero_point_mag,
            radec_list,
        ],
        names=(
            "time_series_images",
            "psf_kernel",
            "obs_time",
            "expo_time",
            "zero_point",
            "calexp_center",
        ),
    )
    return table_data


def multiple_dp0_time_series_images_data(
    butler, center_coords_list, radius="0.034", band="i", size=101, output_file=None
):
    """Creates multiple time series cutouts and associated meta data from dp0
    data. Here, multiple means time series cutouts at multiple sky location.
    Using this function one can produce more than one time series cutout based
    on how many variable lenses he/she want to inject.

    :param butler: butler object
    :param center_coord: list of coordinate point around which we need
        to create time series images.
    :param radius: radius for query
    :param band: imaging band
    :param size: cutout size of images
    :return: List of astropy table containg time series images and other
        information. If output_file path is provided, it saves list of
        these astropy table in fits file with the given name.
    """
    expo_data_list = []
    for center_coords in center_coords_list:
        time_series_data = dp0_time_series_images_data(
            butler, center_coords, radius=radius, band=band, size=size
        )
        if output_file is None:
            expo_data_list.append(time_series_data)
        else:
            first_table = not os.path.exists(output_file)
            if first_table:
                time_series_data.write(output_file, overwrite=True)
                first_table = False
            else:
                fits_append_table(output_file, time_series_data)
    if len(time_series_data) > 1:
        return expo_data_list
    return None


def measure_noise_level_in_RSP_coadd(RSP_coadd, N_pixels, plot=False):
    np.random.seed(1)
    """Function to measure the noise level within a central square aperture of
    an RSP coadd. The noise level could vary between coadds so this should be
    measured on a coadd-by-coadd basis. This is done by fitting a half-norm
    distribution to the negative values in the coadd and then generating a
    large number of random noise realisations from this distribution. The
    maximum flux level (i.e. the aperture flux above which the image is said to
    contain a central source) is then calculated as the 2-sigma limit of the
    sum of the aperture flux in these realisations.

    :param RSP_coadd: .npy array, the RSP coadd image (this should be
        large to ensure the noise level is accurately measured). This
        could also be a 3D array of many individual (random) cutouts.
    :param N_central_pixels: int, size of (square) aperture within which
        to determine the presence/absence of a central source.
    :param plot: bool: Whether to plot the gaussian fits to the noise
        level
    :return: float, 2-sigma flux level in the aperture above which the
        image is said to contain a central source.
    """
    # Select the negative pixel values from the coadd (positive values are excluded to remove the effect of bright sources):
    negative_values = -RSP_coadd.flatten()[RSP_coadd.flatten() < 0]
    # Fitting a half-norm distribution to these pixel values:
    halfnorm0, halfnorm1 = halfnorm.fit(negative_values)
    if plot:
        X_plot = np.linspace(0.2, 0, 100)
        X_plot_full = np.linspace(-0.2, 0.2, 100)
        plt.hist(
            RSP_coadd.flatten(),
            density=True,
            bins=np.linspace(-0.2, 0.2, 100),
            label="Coadd pixel values",
        )
        plt.plot(
            -X_plot,
            0.5 * halfnorm.pdf(X_plot, halfnorm0, halfnorm1),
            label="Half Gaussian",
        )
        plt.plot(
            X_plot_full,
            norm.pdf(X_plot_full, halfnorm0, halfnorm1),
            label="Full Gaussian",
        )
        plt.legend()
        plt.show()
    # Generate 1e+4 realisations of the noise level in the central aperture, and find the summed aperture flux in each:
    rand_norm_array = (
        norm(halfnorm0, halfnorm1)
        .rvs(size=(N_pixels, N_pixels, 10000))
        .sum(axis=0)
        .sum(axis=0)
    )
    # Returns the 2-sigma limit of the aperture fluxes in these realisations:
    return np.mean(rand_norm_array) + 2 * np.std(rand_norm_array)


class retrieve_DP0_coadds_from_Rubin_Science_Platform:
    """Class to retrieve cutouts of DP0.2 coadds, variance maps, PSF arrays and
    exposure maps from the Rubin Science Platform.

    Cutouts of size cutout_size are generated, with the number of
    cutouts per coadd specified by n_im_per_coadd.
    """

    def __init__(
        self,
        butler,
        cutout_size=201,
        n_im_per_coadd=10,
        good_seeing_only=False,
        ra=None,
        dec=None,
    ):
        """
        :param butler: butler object
        :param cutout_size: int, size of the cutout (in pixels) to be generated
        :param n_im_per_coadd: int, number of cutouts to be generated per coadd
        :param plot: bool, whether to plot the cutouts
        :param good_seeing_only: bool, whether to use the goodSeeingCoadd (True) or deepCoadd (False) data products. The goodSeeingCoadd only use the  top one-third best seeing exposures, whereas deepCoadd uses all of them.
        :param ra (optional): float, RA of the central point of the cutout
        :param dec (optional): float, Dec of the central point of the cutout
        """
        assert (ra is None and dec is None) or (
            ra is not None and dec is not None
        )  # Either both ra and dec must be specified or neither.
        if ra is None or dec is None:
            ra_dec_list = random_ra_dec(55, 70, -43, -30, 1)  # Retrieve random RA/Dec
            self.ra = ra_dec_list[0]
            self.dec = ra_dec_list[1]
        else:
            self.ra = ra
            self.dec = dec
        self.butler = butler
        self.skymap = self.butler.get("skyMap")
        self.cutout_size = cutout_size
        self.n_im_per_coadd = n_im_per_coadd
        self.good_seeing_only = good_seeing_only

    def crop_center(self, img, cropx, cropy):
        """Function to crop to the center of an image to specified size
        cropy,cropx :param img: 2D numpy array, the image to be cropped :param
        cropx: int, size of the cropped image in the x-direction :param cropy:
        int, size of the cropped image in the y-direction :return: 2D numpy
        array, the cropped image."""
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    def retrieve_tract_patch(self):
        """Adapted from DC2_cutout (above) Retrieves the tract & patch
        information of the coadd image."""
        self.point = geom.SpherePoint(self.ra, self.dec, geom.degrees)
        self.cutoutSize = geom.ExtentI(self.cutout_size, self.cutout_size)
        self.tractInfo = self.skymap.findTract(self.point)
        patchInfo = self.tractInfo.findPatch(self.point)
        self.tract = self.tractInfo.tract_id
        self.patch = patchInfo.getSequentialIndex()

    def retrieve_coadd_files(self):
        """Adapted from lens_inejection_fast (above) This generates cutouts of
        the coadd, exposure and variance maps.

        The cutout size is specified by cutout_size during
        initialisation.

        :return: 1) Full coadd image, 2) full exposure map image (in
            units of N. exposures), 3) full variance map image 4) list
            of cutout bounding boxes, 5) list of cutout centres
        """
        coaddId_i = {"tract": self.tract, "patch": self.patch, "band": "i"}
        if self.good_seeing_only:
            coadd_i = self.butler.get("goodSeeingCoadd", dataId=coaddId_i)
            coadd_exp_i = self.butler.get("goodSeeingCoadd_nImage", dataId=coaddId_i)
        else:
            coadd_i = self.butler.get("deepCoadd", dataId=coaddId_i)
            coadd_exp_i = self.butler.get("deepCoadd_nImage", dataId=coaddId_i)
        coadd_var_i = coadd_i.getVariance()
        bbox_coadd = coadd_i.getBBox()
        xmin, ymin = bbox_coadd.getBegin()
        xmax, ymax = bbox_coadd.getEnd()
        # Not centering the cutout on pixels close to the edge of the coadd:
        x_center = np.random.randint(xmin + 150, xmax - 150, self.n_im_per_coadd)
        y_center = np.random.randint(ymin + 150, ymax - 150, self.n_im_per_coadd)
        xbox_min = x_center - ((self.cutout_size - 1) / 2)
        xbox_max = x_center + ((self.cutout_size - 1) / 2)
        ybox_min = y_center - ((self.cutout_size - 1) / 2)
        ybox_max = y_center + ((self.cutout_size - 1) / 2)
        bbox_cutout_list = []
        cutout_center_list = []
        for n_cutouts in range(len(x_center)):
            bbox_cutout_i = geom.Box2I(
                geom.Point2I(xbox_min[n_cutouts], ybox_min[n_cutouts]),
                geom.Point2I(xbox_max[n_cutouts], ybox_max[n_cutouts]),
            )
            cutout_centre_i = geom.Point2D(
                0.5 * (xbox_min[n_cutouts] + xbox_max[n_cutouts]),
                0.5 * (ybox_min[n_cutouts] + ybox_max[n_cutouts]),
            )
            bbox_cutout_list.append(bbox_cutout_i)
            cutout_center_list.append(cutout_centre_i)
        return coadd_i, coadd_exp_i, coadd_var_i, bbox_cutout_list, cutout_center_list

    def retrieve_arrays(self):
        """Adapted from cutout_image_psf_kernel (above) This function retrieves
        the coadd images, exposure maps, PSF arrays and variance maps for the
        specified position.

        These arrays are cropped to the specified size, with the
        exception of the PSF array, which is always 57x57.

        :return: 1) list of cutouts, 2) list of exposure maps, 3) list
            of PSF arrays, 4) list of variance maps, 5) uncropped coadd
            image, 6) uncropped variance map
        """
        self.retrieve_tract_patch()
        coadd_im, coadd_exp, var_im, bbox_cutout_list, cutout_center_list = (
            self.retrieve_coadd_files()
        )
        psf = coadd_im.getPsf()
        bbox = coadd_im.getBBox()
        xmin, ymin = bbox.getBegin()
        xmax, ymax = bbox.getEnd()
        # calibFluxRadius = 12
        psf_list = []
        cutout_list = []
        cutout_exp_list = []
        cutout_var_list = []
        # Cropping the arrays to specified size:
        for n_cutouts in range(len(bbox_cutout_list)):
            bbox_cutout_i = bbox_cutout_list[n_cutouts]
            spt_cutout_i = cutout_center_list[n_cutouts]
            cutout_image = coadd_im[bbox_cutout_i]
            cutout_exp = coadd_exp[bbox_cutout_i]
            cutout_var = var_im[bbox_cutout_i]
            psfArr = psf.computeKernelImage(spt_cutout_i).array
            if psfArr.shape != (57, 57):
                psfArr = self.crop_center(psfArr, 57, 57)
            # Not currently applying an aperture correction to the PSF.
            # apCorr = psf.computeApertureFlux(calibFluxRadius, spt_cutout_i)
            # psfArr /= apCorr
            psf_list.append(psfArr)
            cutout_list.append(cutout_image.image.array)
            cutout_exp_list.append(cutout_exp.array)
            cutout_var_list.append(cutout_var.array)
        return cutout_list, cutout_exp_list, psf_list, cutout_var_list, coadd_im, var_im

    def save_arrays(self, foldername, prefix):
        """The generated cutouts are then saved as .h5 files.

        The cutouts are saved as 3D arrays, with the first dimension
        corresponding to the number of cutouts.

        :param foldername: str, name of the folder in which to save the
            files. The folder is generatred if it doesn't exist already.
        :param prefix: str, prefix for the file names (e.g. 0,1,2,3 if
            generating sets of cutouts from different coadds)
        :return: 1) list of cutouts, 2) list of exposure maps, 3) list
            of PSF arrays, 4) list of variance maps, 5) uncropped coadd
            image, 6) uncropped variance map
        """
        (
            cutout_list,
            cutout_exp_list,
            psf_list,
            cutout_var_list,
            full_coadd,
            full_var,
        ) = self.retrieve_arrays()
        cutout_list = np.array(cutout_list)
        cutout_exp_list = np.array(cutout_exp_list)
        cutout_var_list = np.array(cutout_var_list)
        psf_list = np.array(psf_list)
        # Generates the folder if it does not exist:
        # if not os.path.isdir(foldername):
        #     os.mkdir(foldername)
        with h5py.File(foldername + f"/{prefix}_image_data.h5", "w") as hf:
            hf.create_dataset(
                "data",
                data=cutout_list,
                compression="gzip",
                maxshape=(None, cutout_list.shape[1], cutout_list.shape[2]),
            )
        with h5py.File(foldername + f"/{prefix}_var_data.h5", "w") as hf:
            hf.create_dataset(
                "data",
                data=cutout_var_list,
                compression="gzip",
                maxshape=(None, cutout_var_list.shape[1], cutout_var_list.shape[2]),
            )
        with h5py.File(foldername + f"/{prefix}_Nexp_data.h5", "w") as hf:
            hf.create_dataset(
                "data",
                data=cutout_exp_list,
                compression="gzip",
                maxshape=(None, cutout_exp_list.shape[1], cutout_exp_list.shape[2]),
            )
        with h5py.File(foldername + f"/{prefix}_psf_data.h5", "w") as hf:
            hf.create_dataset(
                "data",
                data=psf_list,
                compression="gzip",
                maxshape=(None, psf_list.shape[1], psf_list.shape[2]),
            )
        return (
            cutout_list,
            cutout_exp_list,
            psf_list,
            cutout_var_list,
            full_coadd.image.array,
            full_var.array,
        )


# Define the subquery function
def fetch_DP0_galaxies_from_Rubin_Science_Platform(
    service=rsp_tap,
    ra_min=71.875,
    ra_max=-28.125,
    dec_min=75.0,
    dec_max=-25.0,
):
    """Extracts galaxies from the DP0.2 Object catalog that match the truth
    catalog within a specified region of the sky, defined by right ascension
    (RA) and declination (Dec) ranges.

    : param service : str, the TAP (Table Access Protocol) service endpoint used for querying
        the DP0.2 Object catalog.

    : param ra_min : float, the minimum right ascension (RA) of the required region, in degrees.

    : param ra_max : float, the maximum right ascension (RA) of the required region, in degrees.

    : param dec_min : float, the minimum declination (Dec) of the required region, in degrees.

    : param dec_max : float, the maximum declination (Dec) of the required region, in degrees.

    : returns: astropy.table.table.Table, A pandas daataframe containing the results of the ADQL query,
        including the extracted galaxy data within the specified sky region.
    """

    assert service is not None

    query = f"""
    SELECT mt.id_truth_type AS mt_id_truth_type,
           mt.match_objectId AS mt_match_objectId,
           ts.ra AS ts_ra,
           ts.dec AS ts_dec,
           ts.truth_type AS ts_truth_type,
           ts.mag_r AS ts_mag_r,
           ts.is_pointsource AS ts_is_pointsource,
           ts.redshift AS ts_redshift,
           
           ts.flux_u AS ts_flux_u,
           ts.flux_g AS ts_flux_g,
           ts.flux_r AS ts_flux_r,
           ts.flux_i AS ts_flux_i,
           ts.flux_z AS ts_flux_z,
           ts.flux_y AS ts_flux_y,
           
           obj.coord_ra AS obj_coord_ra,
           obj.coord_dec AS obj_coord_dec,
           obj.refExtendedness AS obj_refExtendedness,
           obj.refBand AS obj_refband,

           obj.u_cModel_flag AS obj_u_cModel_flag,
           obj.u_psfFlux_flag AS obj_u_psfFlux_flag,
           obj.u_cModelFlux AS obj_u_cModelFlux,
           obj.u_cModelFluxErr AS obj_u_cModelFluxerr,
           obj.u_psfFlux AS obj_u_psfFlux,
           obj.u_psfFluxErr AS obj_u_psfFluxerr,
           obj.u_bdFluxB AS obj_u_bdFluxB,
           obj.u_bdFluxBErr AS obj_u_bdFluxBerr,
           obj.u_bdFluxD AS obj_u_bdFluxD,
           obj.u_bdFluxDErr AS obj_u_bdFluxDerr,

           obj.g_cModel_flag AS obj_g_cModel_flag,
           obj.g_psfFlux_flag AS obj_g_psfFlux_flag,
           obj.g_cModelFlux AS obj_g_cModelFlux,
           obj.g_cModelFluxErr AS obj_g_cModelFluxerr,
           obj.g_psfFlux AS obj_g_psfFlux,
           obj.g_psfFluxErr AS obj_g_psfFluxerr,
           obj.g_bdFluxB AS obj_g_bdFluxB,
           obj.g_bdFluxBErr AS obj_g_bdFluxBerr,
           obj.g_bdFluxD AS obj_g_bdFluxD,
           obj.g_bdFluxDErr AS obj_g_bdFluxDerr,

           obj.r_cModel_flag AS obj_r_cModel_flag,
           obj.r_psfFlux_flag AS obj_r_psfFlux_flag,
           obj.r_cModelFlux AS obj_r_cModelFlux,
           obj.r_cModelFluxErr AS obj_r_cModelFluxerr,
           obj.r_psfFlux AS obj_r_psfFlux,
           obj.r_psfFluxErr AS obj_r_psfFluxerr,
           obj.r_bdFluxB AS obj_r_bdFluxB,
           obj.r_bdFluxBErr AS obj_r_bdFluxBerr,
           obj.r_bdFluxD AS obj_r_bdFluxD,
           obj.r_bdFluxDErr AS obj_r_bdFluxDerr,

           obj.i_cModel_flag AS obj_i_cModel_flag,
           obj.i_psfFlux_flag AS obj_i_psfFlux_flag,
           obj.i_cModelFlux AS obj_i_cModelFlux,
           obj.i_cModelFluxErr AS obj_i_cModelFluxerr,
           obj.i_psfFlux AS obj_i_psfFlux,
           obj.i_psfFluxErr AS obj_i_psfFluxerr,
           obj.i_bdFluxB AS obj_i_bdFluxB,
           obj.i_bdFluxBErr AS obj_i_bdFluxBerr,
           obj.i_bdFluxD AS obj_i_bdFluxD,
           obj.i_bdFluxDErr AS obj_i_bdFluxDerr,

           obj.z_cModel_flag AS obj_z_cModel_flag,
           obj.z_psfFlux_flag AS obj_z_psfFlux_flag,
           obj.z_cModelFlux AS obj_z_cModelFlux,
           obj.z_cModelFluxErr AS obj_z_cModelFluxerr,
           obj.z_psfFlux AS obj_z_psfFlux,
           obj.z_psfFluxErr AS obj_z_psfFluxerr,
           obj.z_bdFluxB AS obj_z_bdFluxB,
           obj.z_bdFluxBErr AS obj_z_bdFluxBerr,
           obj.z_bdFluxD AS obj_z_bdFluxD,
           obj.z_bdFluxDErr AS obj_z_bdFluxDerr,

           obj.y_cModel_flag AS obj_y_cModel_flag,
           obj.y_psfFlux_flag AS obj_y_psfFlux_flag,
           obj.y_cModelFlux AS obj_y_cModelFlux,
           obj.y_cModelFluxErr AS obj_y_cModelFluxerr,
           obj.y_psfFlux AS obj_y_psfFlux,
           obj.y_psfFluxErr AS obj_y_psfFluxerr,
           obj.y_bdFluxB AS obj_y_bdFluxB,
           obj.y_bdFluxBErr AS obj_y_bdFluxBerr,
           obj.y_bdFluxD AS obj_y_bdFluxD,
           obj.y_bdFluxDErr AS obj_y_bdFluxDerr,

           obj.shape_xx AS obj_shape_xx,
           obj.shape_xy AS obj_shape_xy,
           obj.shape_yy AS obj_shape_yy,
           
           obj.u_bdReB AS obj_u_bdReB,
           obj.g_bdReB AS obj_g_bdReB,
           obj.r_bdReB AS obj_r_bdReB,
           obj.i_bdReB AS obj_i_bdReB,
           obj.z_bdReB AS obj_z_bdReB,
           obj.y_bdReB AS obj_y_bdReB,

           obj.u_bdReD AS obj_u_bdReD,
           obj.g_bdReD AS obj_g_bdReD,
           obj.r_bdReD AS obj_r_bdReD,
           obj.i_bdReD AS obj_i_bdReD,
           obj.z_bdReD AS obj_z_bdReD,
           obj.y_bdReD AS obj_y_bdReD
           
    FROM dp02_dc2_catalogs.MatchesTruth AS mt
    JOIN dp02_dc2_catalogs.TruthSummary AS ts ON mt.id_truth_type = ts.id_truth_type
    JOIN dp02_dc2_catalogs.Object AS obj ON mt.match_objectId = obj.objectId
    WHERE CONTAINS(POINT('ICRS', obj.coord_ra, obj.coord_dec), 
                   POLYGON('ICRS', {ra_min}, {dec_min}, 
                                  {ra_min}, {dec_max}, 
                                  {ra_max}, {dec_max}, 
                                  {ra_max}, {dec_min})) = 1
    AND ts.truth_type = 1
    AND obj.refExtendedness = 1
    AND obj.u_cModelFlux > 0
    AND obj.g_cModelFlux > 0
    AND obj.r_cModelFlux > 0
    AND obj.i_cModelFlux > 0
    AND obj.z_cModelFlux > 0
    AND obj.y_cModelFlux > 0
    AND obj.u_cModel_flag = 0
    AND obj.g_cModel_flag = 0
    AND obj.r_cModel_flag = 0
    AND obj.i_cModel_flag = 0
    AND obj.z_cModel_flag = 0
    AND obj.y_cModel_flag = 0
    AND obj.detect_isPrimary = 1
    """

    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR"])
    print("Job phase is", job.phase)

    # Fetch and return results if the job is completed
    if job.phase == "COMPLETED":
        results = job.fetch_result().to_table()
        print(f"Number of results: {len(results)}")
        return results
    else:
        print("Job failed with error phase.")
        return None
