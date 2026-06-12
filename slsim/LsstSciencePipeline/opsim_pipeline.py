import numpy as np
from astropy.table import Table
import lenstronomy.Util.util as util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.data_util as data_util


def opsim_time_series_images_data(
    ra_list,
    dec_list,
    obs_strategy,
    MJD_min=60000,
    MJD_max=64500,
    num_pix=101,
    moffat_beta=3.1,
    readout_noise=10,
    delta_pix=0.2,
    print_warning=True,
    opsim_path=None,
):
    """Creates time series data from opsim database.

    :param ra_list: a list of ra points (in degrees) from objects we want to collect
        observations for
    :param dec_list: a list of dec points (in degrees) from objects we want to collect
        observations for
    :param obs_strategy: version of observing strategy corresponding to opsim database.
        for example "baseline_v3.0_10yrs" (string)
    :param MJD_min: minimum MJD for the observations
    :param MJD_max: maximum MJD for the observations
    :param num_pix: cutout size of images (in pixels)
    :param moffat_beta: power index of the moffat psf kernel
    :param readout_noise: noise added per readout
    :param delta_pix: size of pixel in units arcseonds
    :param print_warning: if True, prints a warning of coordinates outside of the LSST
        footprint
    :param opsim_path: optional: provide a path to the opsim database.
        if None: use "../data/OpSim_database/" + obs_strategy + ".db" as default path.
    :return: a list of astropy tables containing observation information for each
        coordinate
    """

    # Import OpSimSummaryV2
    try:
        import opsimsummaryv2 as op
    except ImportError:
        raise ImportError(
            "Users need to have OpSimSummaryV2 installed (https://github.com/bastiencarreres/OpSimSummaryV2)"
        )

    # Initialise OpSimSummaryV2 with opsim database
    if opsim_path is None:
        opsim_path = "../data/OpSim_database/" + obs_strategy + ".db"
    try:
        OpSimSurv = op.OpSimSurvey(opsim_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "File not found: "
            + opsim_path
            + ". Input variable 'obs_strategy' should correspond to the name of an opsim database saved in the folder ../data/OpSim_database"
        )

    # Collect observations that cover the coordinates in ra_list and dec_list
    gen = OpSimSurv.get_obs_from_coords(
        ra_list,
        dec_list,
        is_deg=True,
        formatobs=True,
        keep_keys=["visitExposureTime", "seeingFwhmGeom", "fieldRA", "fieldDec"],
    )

    table_data_list = []

    # Loop through all coordinates and compute the table_data
    for i in range(len(ra_list)):

        # Collect the next observation sequence from the opsim generator
        seq = next(gen)
        seq = seq.sort_values(by=["expMJD"])

        # Check if the coordinates are in the opsim LSST footprint
        opsim_ra = np.mean(seq["fieldRA"])
        opsim_dec = np.mean(seq["fieldDec"])

        if np.isnan(opsim_ra) or np.isnan(opsim_dec):
            if print_warning:
                print(
                    f"Coordinate ({ra_list[i]}, {dec_list[i]}) is not in the LSST footprint. This entry is skipped."
                )
            continue

        # Get the relevant properties from opsim
        obs_time = np.array(seq["expMJD"])

        # Only give the observations between MJD_min and MJD_max
        mask = (obs_time > MJD_min) & (obs_time < MJD_max)
        obs_time = obs_time[mask]

        expo_time = np.array(seq["visitExposureTime"])[mask]
        bandpass = np.array(seq["BAND"])[mask]
        zero_point_mag = np.array(seq["ZPT"])[mask]
        sky_brightness = np.array(seq["SKYSIG"])[mask] ** 2 / (delta_pix**2 * expo_time)
        psf_fwhm = np.array(seq["seeingFwhmGeom"])[mask]
        # Question: use 'FWHMeff' or 'seeingFwhmGeom' for the psf?

        radec_list = [(ra_list[i], dec_list[i])] * len(obs_time)

        # Create a Moffat psf kernel for each epoch

        psf_kernels = []

        for psf in psf_fwhm:
            psf_kernel = kernel_util.kernel_moffat(
                num_pix=num_pix, delta_pix=delta_pix, fwhm=psf, moffat_beta=moffat_beta
            )
            psf_kernel = util.array2image(psf_kernel)

            psf_kernels.append(psf_kernel)

        psf_kernels = np.array(psf_kernels)

        # Calculate background noise
        bkg_noise = data_util.bkg_noise(
            readout_noise, expo_time, sky_brightness, delta_pix, num_exposures=1
        )

        table_data = Table(
            [
                bkg_noise,
                psf_kernels,
                obs_time,
                expo_time,
                zero_point_mag,
                psf_fwhm,
                radec_list,
                bandpass,
            ],
            names=(
                "bkg_noise",
                "psf_kernel",
                "obs_time",
                "expo_time",
                "zero_point",
                "psf_fwhm",
                "calexp_center",
                "band",
            ),
        )

        table_data_list.append(table_data)
    return table_data_list
