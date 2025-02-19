import numpy as np


def get_errors_Poisson(app_mag, zeropoint, exptime):
    """This function provides rough photometric errors using the photometric
    magnitudes, assuming only the statistical error due to Poisson noise in the
    counts. Since, the photometric errors are not known in the LSST mock
    catalogs, these are just very rough estimates.

    It requires the apparent magnitude, photometric zeropoint and the exposure time.

    input_params:

    app_mag: the apparent magnitude of the object.
    type: float or 1-D array.

    zeropoint: the photometric zeropoint in the given band.
    type: float or 1-D array.
    Note that photometric zeropoint will be different for different bands

    exptime: the exposure time for the given band (in seconds)
    type: float
    """

    ins_mag = app_mag - zeropoint

    # convert instrumental magnitude to the flux
    flux = 10 ** (-0.4 * ins_mag)
    counts_per_sec = flux
    total_counts = counts_per_sec * exptime
    counts_err = np.sqrt(total_counts)

    flux_err = counts_err / exptime
    ins_mag_err = (2.5 * flux_err) / (flux * np.log(10))
    app_mag_err = ins_mag_err
    return app_mag_err
