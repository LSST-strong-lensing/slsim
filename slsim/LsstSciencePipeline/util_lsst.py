import numpy as np
from astropy.table import Column
from slsim.ImageSimulation.image_simulation import lens_image_series, lens_image
from slsim.Util.param_util import (
    fits_append_table,
    convert_mjd_to_days,
    transient_event_time_mjd,
    flux_error_to_magnitude_error,
    magnitude_to_amplitude,
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
        of dp0 data.
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


def transient_data_with_cadence(
    lens_class,
    exposure_data,
    noise=True,
    symmetric=False,
    pix_scale=0.2,
    random_seed=None,
):
    """Puts lensed transient into the provided cadence. For LSST, this will be
    cadence from opsim.

    :param lens_class: Lens() object
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
    :param noise: Boolean. If True, a gaussian noise is added to the
        lightcurve flux.
    :param symmetric: Boolean. If True, a symmetric error on magnitude
        is provided.
    :param pixscale: pixel scale of the observing instrument.
    :return: Astropy table of lightcurve and exposure information of dp0
        data. The table contains: Observation time in days, lens id,
        magnitude of each image and associated errors, lens image. If
        the lens system produces fewer than four images, the missing
        magnitudes and errors are filled with -1.
    """
    copied_exposure_data = exposure_data.copy()
    min_mjd = min(copied_exposure_data["obs_time"])
    max_mjd = max(copied_exposure_data["obs_time"])

    min_lc_time, max_lc_time = (
        min(lens_class.source(0)._source._lightcurve_time),
        max(lens_class.source(0)._source._lightcurve_time),
    )

    start_mjd_time = transient_event_time_mjd(min_mjd, max_mjd, random_seed=random_seed)

    copied_exposure_data["obs_time_in_days"] = convert_mjd_to_days(
        copied_exposure_data["obs_time"], start_mjd_time
    )

    copied_exposure_data = copied_exposure_data[
        (copied_exposure_data["obs_time_in_days"] >= min_lc_time - 50)
        & (copied_exposure_data["obs_time_in_days"] <= max_lc_time)
    ]

    ra_dec = copied_exposure_data["calexp_center"][0]
    lens_id = lens_class.generate_id(ra=ra_dec[0].degree, dec=ra_dec[1].degree)

    num_images = lens_class.image_number[0]
    mag_images = {f"mag_image_{i+1}": [] for i in range(num_images)}
    mag_errors = {
        f"mag_error_image_{i+1}_{err}": []
        for i in range(num_images)
        for err in ["low", "high"]
    }

    for exposure in copied_exposure_data:
        obs_time = exposure["obs_time_in_days"]
        magnitudes = lens_class.point_source_magnitude(
            band=exposure["band"], lensed=True, time=obs_time
        )[0]

        # Compute noise
        bkg_noise, fwhm = exposure["bkg_noise"], exposure["psf_fwhm"]
        N_pix = np.pi * (2 * fwhm) ** 2 / (pix_scale**2)
        sigma_noise_total = bkg_noise / np.sqrt(N_pix)

        for i in range(num_images):
            amplitude = magnitude_to_amplitude(magnitudes[i], exposure["zero_point"])
            total_counts = amplitude * exposure["expo_time"]
            poisson_noise = np.sqrt(total_counts)
            flux_err = np.sqrt(sigma_noise_total**2 + poisson_noise**2)

            mag_realiz, mag_err_low, mag_err_high = flux_error_to_magnitude_error(
                amplitude,
                flux_err,
                exposure["zero_point"],
                noise=noise,
                symmetric=symmetric,
            )

            mag_images[f"mag_image_{i+1}"].append(mag_realiz)
            mag_errors[f"mag_error_image_{i+1}_low"].append(mag_err_low)
            mag_errors[f"mag_error_image_{i+1}_high"].append(mag_err_high)

    # Fill missing values for systems with <4 images
    for i in range(num_images, 4):
        mag_images[f"mag_image_{i+1}"] = [-1.0] * len(copied_exposure_data)
        mag_errors[f"mag_error_image_{i+1}_low"] = [-1.0] * len(copied_exposure_data)
        mag_errors[f"mag_error_image_{i+1}_high"] = [-1.0] * len(copied_exposure_data)

    for key in list(mag_images.keys()) + list(mag_errors.keys()):
        mag_images_or_errors = mag_images if key in mag_images else mag_errors
        mag_images_or_errors[key] = np.array(
            mag_images_or_errors[key], dtype=float
        ).reshape(-1)

    # Create and add columns to the table
    copied_exposure_data.add_columns(
        [Column(name="lens_id", data=[lens_id] * len(copied_exposure_data))]
        + [Column(name=name, data=data) for name, data in mag_images.items()]
        + [Column(name=name, data=data) for name, data in mag_errors.items()]
    )

    return copied_exposure_data


def extract_lightcurves_in_different_bands(transient_data_table):
    """Extract lightcurves and images in different bands from the given
    catalog. This a function written to read data table from
    transient_data_with_cadence function above.

    :param transient_data_table: An astropy table containing lightcurves
        in a certain cadence. This table must contain magnitude and
        corresponding errors. The column name for magnitude should be
        "mag_image_n", and column names for the error should be
        "mag_error_image_n_low" and "mag_error_image_n_high", where n
        could be 1, 2, 3, or 4.
    :return: A dictionary of magnitudes, associated errors, observation
        times, structured by band.
    """
    table = transient_data_table
    # Extract unique bands
    bands = np.unique(table["band"])

    # Initialize dictionaries to hold magnitudes, errors, observation times, and
    # optionally image lists
    magnitudes = {f"mag_image_{i}": {band: [] for band in bands} for i in range(1, 5)}
    errors_low = {
        f"mag_error_image_{i}_low": {band: [] for band in bands} for i in range(1, 5)
    }
    errors_high = {
        f"mag_error_image_{i}_high": {band: [] for band in bands} for i in range(1, 5)
    }
    obs_time = {band: [] for band in bands}

    # Populate dictionaries with magnitudes, errors, and optionally image lists
    # corresponding to each band
    for band in bands:
        # Filter rows that correspond to the current band
        band_data = table[table["band"] == band]
        obs_time[band] = band_data["obs_time"].tolist()

        for i in range(1, 5):
            mag_col = f"mag_image_{i}"
            err_low_col = f"mag_error_image_{i}_low"
            err_high_col = f"mag_error_image_{i}_high"

            if mag_col in band_data.colnames and np.any(band_data[mag_col] != -1):
                # Only proceed with the column if the magnitude is not -1
                magnitudes[mag_col][band] = band_data[mag_col].tolist()
                errors_low[err_low_col][band] = band_data[err_low_col].tolist()
                errors_high[err_high_col][band] = band_data[err_high_col].tolist()

    result = {
        "magnitudes": magnitudes,
        "errors_low": errors_low,
        "errors_high": errors_high,
        "obs_time": obs_time,
    }

    return result
