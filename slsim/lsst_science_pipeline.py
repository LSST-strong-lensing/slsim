import numpy as np
import lsst.geom as geom

# Source injection
from lsst.pipe.tasks.insertFakes import _add_fake_sources
import galsim
from astropy.table import Table, vstack
from slsim.image_simulation import sharp_image
from scipy.signal import convolve2d
from scipy import interpolate
from slsim.image_simulation import point_source_coordinate_properties

"""
This module provides necessary functions to inject lenses to the dp0 data. For this, it 
uses some of the packages provided by the LSST Science Pipeline.
"""


def DC2_cutout(ra, dec, num_pix, butler, band):
    """Draws a cutout from the DC2 data based on the given ra, dec pair. For this one
    needs to provide a butler to this function. To initiate Butler, you need to specify
    data configuration and collection of the data.

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
    """Chooses a random lens from the lens population and injects it to a DC2 cutout
    image. For this one needs to provide a butler to this function. To initiate Butler,
    you need to specify data configuration and collection of the data.

    :param lens_pop: lens population from slsim
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param lens_cut: list of criteria for lens selection
    :param flux: flux need to be asigned to the lens image. It sould be None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lens in r-band, DC2 cutout image in
        r-band, cutout image with injected lens in r, g , and i band
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


def lens_inejection_fast(
    lens_pop,
    num_pix,
    delta_pix,
    butler,
    ra,
    dec,
    num_cutout_per_patch=10,
    lens_cut=None,
    flux=None,
):
    """Chooses a random lens from the lens population and injects it to a DC2 cutout
    image. For this one needs to provide a butler to this function. To initiate Butler,
    you need to specify data configuration and collection of the data.

    :param lens_pop: lens population from slsim
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param num_cutout_per_patch: number of cutout image drawn per patch
    :param lens_cut: list of criteria for lens selection
    :param flux: flux need to be asigned to the lens image. It sould be None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lens in r-band, DC2 cutout image in
        r-band, cutout image with injected lens in r, g , and i band
    """

    if lens_cut is None:
        kwargs_lens_cut = {}
    else:
        kwargs_lens_cut = lens_cut

    rgb_band_list = ["r", "g", "i"]
    skymap = butler.get("skyMap")
    point = geom.SpherePoint(ra, dec, geom.degrees)
    # cutoutSize = geom.ExtentI(num_pix, num_pix)

    tractInfo = skymap.findTract(point)
    patchInfo = tractInfo.findPatch(point)
    my_tract = tractInfo.tract_id
    my_patch = patchInfo.getSequentialIndex()

    coadd = []
    for band in rgb_band_list:
        coaddId = {"tract": my_tract, "patch": my_patch, "band": band}

        coadd.append(butler.get("deepCoadd", dataId=coaddId))

    bbox = coadd[0].getBBox()
    xmin, ymin = bbox.getBegin()
    xmax, ymax = bbox.getEnd()
    wcs = coadd[0].getWcs()

    x_center = np.random.randint(xmin + 150, xmax - 150, num_cutout_per_patch)
    y_center = np.random.randint(ymin + 150, ymax - 150, num_cutout_per_patch)
    xbox_min = x_center - ((num_pix - 1) / 2)
    xbox_max = x_center + ((num_pix - 1) / 2)
    ybox_min = y_center - ((num_pix - 1) / 2)
    ybox_max = y_center + ((num_pix - 1) / 2)

    table = []
    for i in range(len(x_center)):
        lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
        cutout_bbox = geom.Box2I(
            geom.Point2I(xbox_min[i], ybox_min[i]),
            geom.Point2I(xbox_max[i], ybox_max[i]),
        )
        injected_final_image = []
        box_center = []
        cutout_image_list = []
        lens_image = []
        for j in range(len(coadd)):
            lens = sharp_image(
                lens_class=lens_class,
                band=rgb_band_list[j],
                mag_zero_point=27,
                delta_pix=delta_pix,
                num_pix=num_pix,
            )
            cutout_image = coadd[j][cutout_bbox]
            objects = [(geom.Point2D(x_center[i], y_center[i]), lens, delta_pix)]
            final_injected_image = add_object(cutout_image, objects, calibFluxRadius=12)
            center_wcs = wcs.pixelToSky(objects[0][0])
            ra_deg = center_wcs.getRa().asDegrees()
            dec_deg = center_wcs.getDec().asDegrees()

            injected_final_image.append(final_injected_image)
            box_center.append((ra_deg, dec_deg))
            cutout_image_list.append(cutout_image.image.array)
            lens_image.append((final_injected_image - cutout_image.image.array))
        table_1 = Table(
            [
                [lens_image[0]],
                [cutout_image_list[0]],
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
        table.append(table_1)
    lens_catalog = vstack(table)
    return lens_catalog


def multiple_lens_injection(
    lens_pop, num_pix, delta_pix, butler, ra, dec, lens_cut=None, flux=None
):
    """Injects random lenses from the lens population to multiple DC2 cutout images
    using lens_inejection function. For this one needs to provide a butler to this
    function. To initiate Butler, you need to specify data configuration and collection
    of the data.

    :param lens_pop: lens population from slsim
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param ra: ra for a cutout
    :param dec: dec for a cutout
    :param flux: flux need to be asigned to the lens image. It sould be None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lenses in r-band, DC2 cutout images
        in r-band, cutout images with injected lens in r, g , and i band for a given set
        of ra and dec
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
                lens_cut=None,
                flux=None,
            )
        )
    injected_image_catalog = vstack(injected_images)
    return injected_image_catalog


def multiple_lens_injection_fast(
    lens_pop,
    num_pix,
    delta_pix,
    butler,
    ra,
    dec,
    num_cutout_per_patch=10,
    lens_cut=None,
    flux=None,
):
    """Injects random lenses from the lens population to multiple DC2 cutout images
    using lens_inejection_fast function. For this one needs to provide a butler to this
    function. To initiate Butler, you need to specify data configuration and collection
    of the data.

    :param lens_pop: lens population from slsim
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param ra: ra for a cutout
    :param dec: dec for a cutout
    :param flux: flux need to be asigned to the lens image. It sould be None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lenses in r-band, DC2 cutout images
        in r-band, cutout images with injected lens in r, g , and i band for a given set
        of ra and dec
    """
    injected_images = []
    for i in range(len(ra)):
        injected_images.append(
            lens_inejection_fast(
                lens_pop,
                num_pix,
                delta_pix,
                butler,
                ra[i],
                dec[i],
                num_cutout_per_patch,
                lens_cut=None,
                flux=None,
            )
        )
    injected_image_catalog = vstack(injected_images)
    return injected_image_catalog


def add_object(dp0_image, objects, calibFluxRadius=12):
    """Injects a given object in a dp0 cutout image.

    :param dp0_image: cutout image from the dp0 data or any other image
    :param objects: a tuple of point/coordinate where we want to inject the image,
        source image, and pixel scale of source image. Eg. [(point, image, pixel_scale)]
    :param calibFluxRadius: (optional) Aperture radius (in pixels) used to define the
        calibration for thisexposure+catalog. This is used to produce the correct
        instrumental fluxes within the radius. The value should match that of the field
        defined in slot_CalibFlux_instFlux.
    :returns: an image with injected source
    """
    wcs = dp0_image.getWcs()
    psf = dp0_image.getPsf()
    bbox = dp0_image.getBBox()
    pixscale = wcs.getPixelScale(bbox.getCenter()).asArcseconds()
    num_pix_cutout = np.shape(dp0_image.image.array)[0]
    for spt, lens, pix_scale in objects:
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
                "Images with different pixel scale should be combined. Please make"
                "sure that your lens image and dp0 cutout image have compatible pixel"
                "scale."
            )
        else:
            pt = spt
            psfArr = psf.computeKernelImage(pt).array
            apCorr = psf.computeApertureFlux(calibFluxRadius, pt)

            psfArr /= apCorr
            convolved_image = convolve2d(
                lens, psfArr, mode="same", boundary="symm", fillvalue=0.0
            )
            injected_image = np.array(dp0_image.image.array) + np.array(convolved_image)
            return injected_image


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
    """This function extracts psf kernels from the dp0 cutout image at point source
    image positions and deflector position. In the dp0.2 data, psf kernel vary with
    coordinate and can be computed using given psf model.

    :param dp0_image: cutout image from the dp0 data or any other image
    :param lens_class: class object containing all information of the lensing system
        (e.g., Lens())
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :param calibFluxRadius: (optional) Aperture radius (in pixels) used to define the
        calibration for thisexposure+catalog. This is used to produce the correct
        instrumental fluxes within the radius. The value should match that of the field
        defined in slot_CalibFlux_instFlux.
    :returns: Astropy table containing psf kernel at image and deflector positions.
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
