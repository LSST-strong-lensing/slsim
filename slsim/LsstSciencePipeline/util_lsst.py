import numpy as np
from astropy.table import Column
from slsim.image_simulation import lens_image_series
from slsim.Util.param_util import fits_append_table
import os


def variable_lens_injection(
    lens_class, band, num_pix, transform_pix2angle, exposure_data
):
    """Injects variable lens to the dp0 time series data.

    :param lens_class: Lens() object
    :param band: imaging band
    :param num_pix: number of pixels per axis
    :param transform_pix2angle: transformation matrix (2x2) of pixels into coordinate
        displacements
    :param exposure_data: An astropy table of exposure data. It must contain calexp
        images or generated noisy background image (column name should be
        "time_series_images", these images are single exposure images of the same part
        of the sky at different time), magnitude zero point (column name should be
        "zero_point", these are zero point magnitudes for each single exposure images in
        time series image) , psf kernel for each exposure (column name should be
        "psf_kernel", these are pixel psf kernel for each single exposure images in time
        series image), exposure time (column name should be "expo_time", these are
        exposure time for each single exposure images in time series images),
        observation time (column name should be "obs_time", these are observation time
        in days for each single exposure images in time series images)
    :return: Astropy table of injected lenses and exposure information of dp0 data
    """
    # the range of observation time of single exposure images might be outside of the
    # lightcurve time. So, we use random observation time from the lens class lightcurve
    # time to ensure simulation of reasonable images.
    observation_time = np.random.uniform(
        min(lens_class.single_source_class.lightcurve_time),
        max(lens_class.single_source_class.lightcurve_time),
        size=len(exposure_data["obs_time"]),
    )
    observation_time.sort()
    new_obs_time = Column(name="obs_time", data=observation_time)
    exposure_data.replace_column("obs_time", new_obs_time)
    lens_images = lens_image_series(
        lens_class,
        band=band,
        mag_zero_point=exposure_data["zero_point"],
        num_pix=num_pix,
        psf_kernel=exposure_data["psf_kernel"],
        transform_pix2angle=transform_pix2angle,
        exposure_time=exposure_data["expo_time"],
        t_obs=exposure_data["obs_time"],
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
    :param transform_matrices_list: list of transformation matrix (2x2) of pixels into
        coordinate displacements for each exposure
    :param exposure_data: list of astropy table of exposure data for each set of time
        series images. It must contain calexp images or generated noisy background image
        (column name should be "time_series_images", these images are single exposure
        images of the same part of the sky at different time), magnitude zero point
        (column name should be "zero_point", these are zero point magnitudes for each
        single exposure images in time series image) , psf kernel for each exposure
        (column name should be "psf_kernel", these are pixel psf kernel for each single
        exposure images in time series image), exposure time (column name should be
        "expo_time", these are exposure time for each single exposure images in time
        series images), observation time (column name should be "obs_time", these are
        observation time in days for each single exposure images in time series images)
    :param output_file: path to the output FITS file where data will be saved
    :return: list of astropy table of injected lenses and exposure information of dp0
        data for each time series lenses. If output_file path is provided, it saves list
        of these astropy table in fits file with the given name.
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