import numpy as np
from astropy.table import Column
from slsim.image_simulation import lens_image_series, lens_image
from slsim.Util.param_util import (
    fits_append_table,
    convert_mjd_to_days,
    transient_event_time_mjd,
)
import os


def variable_lens_injection(
    lens_class, band, num_pix, transform_pix2angle, exposure_data
):
    """Injects variable lens to the dp0 time series data.

    :param lens_class: Lens() object
    :param band: imaging band
    :param num_pix: number of pixels per axis
    :param transform_pix2angle: transformation matrix (2x2) of pixels
        into coordinate displacements
    :param exposure_data: An astropy table of exposure data. It must
        contain calexp images or generated noisy background image
        (column name should be "time_series_images", these images are
        single exposure images of the same part of the sky at different
        time), magnitude zero point (column name should be "zero_point",
        these are zero point magnitudes for each single exposure images
        in time series image) , psf kernel for each exposure (column
        name should be "psf_kernel", these are pixel psf kernel for each
        single exposure images in time series image), exposure time
        (column name should be "expo_time", these are exposure time for
        each single exposure images in time series images), observation
        time (column name should be "obs_time", these are observation
        time in days for each single exposure images in time series
        images)
    :return: Astropy table of injected lenses and exposure information
        of dp0 data
    """
    ## chose transient starting point randomly.
    start_point_mjd_time = transient_event_time_mjd(
        min(exposure_data["obs_time"]), max(exposure_data["obs_time"])
    )
    ## Convert mjd observation time to days. We should do this because lightcurves are
    #  in the unit of days.
    observation_time = convert_mjd_to_days(
        exposure_data["obs_time"], start_point_mjd_time
    )
    lens_images = lens_image_series(
        lens_class,
        band=band,
        mag_zero_point=exposure_data["zero_point"],
        num_pix=num_pix,
        psf_kernel=exposure_data["psf_kernel"],
        transform_pix2angle=transform_pix2angle,
        exposure_time=exposure_data["expo_time"],
        t_obs=observation_time,
    )

    final_image = []
    for i in range(len(exposure_data["obs_time"])):
        final_image.append(exposure_data["time_series_images"][i] + lens_images[i])
    lens_col = Column(name="lens", data=lens_images)
    final_image_col = Column(name="injected_lens", data=final_image)
    if "lens" in exposure_data.colnames:
        exposure_data.replace_column("lens", lens_col)
    else:
        exposure_data.add_column(lens_col)
    if "injected_lens" in exposure_data.colnames:
        exposure_data.replace_column("injected_lens", final_image_col)
    else:
        exposure_data.add_column(final_image_col)
    return exposure_data


def multiple_variable_lens_injection(
    lens_class_list,
    band,
    num_pix,
    transform_matrices_list,
    exposure_data_list,
    output_file=None,
):
    """Injects multiple variable lenses to multiple dp0 time series data.

    :param lens_class_list: list of Lens() object
    :param band: imaging band
    :param num_pix: number of pixels per axis
    :param transform_matrices_list: list of transformation matrix (2x2)
        of pixels into coordinate displacements for each exposure
    :param exposure_data: list of astropy table of exposure data for
        each set of time series images. It must contain calexp images or
        generated noisy background image (column name should be
        "time_series_images", these images are single exposure images of
        the same part of the sky at different time), magnitude zero
        point (column name should be "zero_point", these are zero point
        magnitudes for each single exposure images in time series image)
        , psf kernel for each exposure (column name should be
        "psf_kernel", these are pixel psf kernel for each single
        exposure images in time series image), exposure time (column
        name should be "expo_time", these are exposure time for each
        single exposure images in time series images), observation time
        (column name should be "obs_time", these are observation time in
        days for each single exposure images in time series images)
    :param output_file: path to the output FITS file where data will be
        saved
    :return: list of astropy table of injected lenses and exposure
        information of dp0 data for each time series lenses. If
        output_file path is provided, it saves list of these astropy
        table in fits file with the given name.
    """
    final_images_catalog = []
    for lens_class, transform_matrices, expo_data in zip(
        lens_class_list, transform_matrices_list, exposure_data_list
    ):
        variable_injected_image = variable_lens_injection(
            lens_class,
            band=band,
            num_pix=num_pix,
            transform_pix2angle=transform_matrices,
            exposure_data=expo_data,
        )
        if output_file is None:
            final_images_catalog.append(variable_injected_image)
        else:
            first_table = not os.path.exists(output_file)
            if first_table:
                variable_injected_image.write(output_file, overwrite=True)
                first_table = False
            else:
                fits_append_table(output_file, variable_injected_image)
    if len(final_images_catalog) > 1:
        return final_images_catalog
    return None


def opsim_variable_lens_injection(
    lens_class, bands, num_pix, transform_pix2angle, exposure_data
):
    """Injects variable lens to the OpSim time series data (1 object).

    :param lens_class: Lens() object
    :param bands: list of imaging bands of interest
    :param num_pix: number of pixels per axis
    :param transform_pix2angle: transformation matrix (2x2) of pixels
        into coordinate displacements
    :param exposure_data: An astropy table of exposure data. One entry
        of table_list_data generated from the
        opsim_time_series_images_data function. It must contain the rms
        of background noise fluctuations (column name should be
        "bkg_noise"), psf kernel for each exposure (column name should
        be "psf_kernel", these are pixel psf kernel for each single
        exposure images in time series image), observation time (column
        name should be "obs_time", these are observation time in days
        for each single exposure images in time series images), exposure
        time (column name should be "expo_time", these are exposure time
        for each single exposure images in time series images),
        magnitude zero point (column name should be "zero_point", these
        are zero point magnitudes for each single exposure images in
        time series image), coordinates of the object (column name
        should be "calexp_center"), these are the coordinates in (ra,
        dec), and the band in which the observation is taken (column
        name should be "band").
    :return: Astropy table of injected lenses and exposure information
        of dp0 data
    """

    ## chose transient starting point randomly.
    start_point_mjd_time = transient_event_time_mjd(
        min(exposure_data["obs_time"]), max(exposure_data["obs_time"])
    )
    final_image = []

    for obs in range(len(exposure_data["obs_time"])):

        exposure_data_obs = exposure_data[obs]
        observation_time = convert_mjd_to_days(
            exposure_data_obs["obs_time"], start_point_mjd_time
        )
        if exposure_data_obs["band"] not in bands:
            continue

        if "bkg_noise" in exposure_data_obs.keys():
            std_gaussian_noise = exposure_data_obs["bkg_noise"]
        else:
            std_gaussian_noise = None

        lens_images = lens_image(
            lens_class,
            band=exposure_data_obs["band"],
            mag_zero_point=exposure_data_obs["zero_point"],
            num_pix=num_pix,
            psf_kernel=exposure_data_obs["psf_kernel"],
            transform_pix2angle=transform_pix2angle,
            exposure_time=exposure_data_obs["expo_time"],
            t_obs=observation_time,
            std_gaussian_noise=std_gaussian_noise,
        )

        final_image.append(lens_images)

    lens_col = Column(name="lens", data=final_image)
    final_image_col = Column(name="injected_lens", data=final_image)

    # Create a new Table with only the bands of interest
    expo_bands = np.array([b for b in exposure_data["band"]])
    mask = np.isin(expo_bands, bands)
    exposure_data_new = exposure_data[mask]

    # if len(exposure_data_new) > 0:
    exposure_data_new.add_columns([lens_col, final_image_col])

    return exposure_data_new
