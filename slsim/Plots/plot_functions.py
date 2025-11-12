import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import random
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from slsim.Microlensing.magmap import MagnificationMap

"""This module contains various plotting definations."""


def create_image_montage_from_image_list(
    num_rows,
    num_cols,
    images,
    time=None,
    band=None,
    image_type="other",
    image_center=None,
):
    """Creates an image montage from an image list.

    :param num_rows: number of images to display horizontally
    :param num_cols: number of images to display vertically
    :param images: list of images
    :param time: array of observation time for point source images. If
        None, considers static case.
    :param band: array of bands corresponding to the observations. If
        None, does not display any information regarding the band.
    :param image_type: type of the provided image. It could be 'dp0' or
        any other name.
    :param image_center: center of the source images.
    :type image_center: array. eg: for two image, it should be like
        np.array([[13.71649063, 13.09556121], [16.69249276,
        17.78106655]])
    :return: image montage of given images.
    """

    # Collect min and max values from all images
    all_min = []
    all_max = []
    for image in images:
        all_min.append(np.min(image))
        all_max.append(np.max(image))
    global_min = min(all_min)
    global_max = max(all_max)

    # If band is one string, extend to list
    if isinstance(band, str):
        band = [band] * len(images)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < len(images):
                image = images[i * num_cols + j]

                if image_type == "dp0":
                    zscale = ZScaleInterval()
                    vmin, vmax = zscale.get_limits(image)
                    axes[i, j].imshow(
                        image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax
                    )
                else:
                    axes[i, j].imshow(
                        image, origin="lower", vmin=global_min, vmax=global_max
                    )
                axes[i, j].axis("off")  # Turn off axis labels
                if time is not None:
                    axes[i, j].text(
                        0.05,
                        0.95,
                        f"Time: {round(time[i * num_cols + j],2)} days",
                        fontsize=10,
                        color="white",
                        verticalalignment="top",
                        horizontalalignment="left",
                        transform=axes[i, j].transAxes,
                    )
                if band is not None:
                    axes[i, j].text(
                        0.05,
                        0.10,
                        f"Band: {band[i * num_cols + j]}",
                        fontsize=10,
                        color="white",
                        verticalalignment="top",
                        horizontalalignment="left",
                        transform=axes[i, j].transAxes,
                    )
                if image_center is not None:
                    for k in range(len(image_center)):
                        axes[i, j].scatter(
                            image_center[k][0],
                            image_center[k][1],
                            marker="*",
                            color="red",
                            s=30,
                        )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    return fig


def plot_montage_of_random_injected_lens(image_list, num, n_horizont=1, n_vertical=1):
    """Creates an image montage of random lenses from the catalog of injected
    lens.

    :param images_list: list of catalog images
    :param n_horizont: number of images to display horizontally
    :param n_vertical: number of images to display vertically
    :param num: length of the injected lens catalog
    :return: image montage of random injected lenses.
    """
    fig, axes = plt.subplots(
        n_vertical, n_horizont, figsize=(n_horizont * 3, n_vertical * 3)
    )
    for i in range(n_horizont):
        for j in range(n_vertical):
            ax = axes[j, i]
            index = random.randint(0, num)
            image = image_list[index]
            ax.imshow(image, aspect="equal", origin="lower")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)

    fig.tight_layout()
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
    )
    return fig


def plot_lightcurves(lightcurve_dict):
    """Plots lightcurves dynamically for all available images across different
    bands.

    :param lightcurve_dict: Dictionary of lightciurves.
         The format of this dictionary should be following:
         lightcurve_dict = {
        "obs_time": {
            "i": [63105.42, 63107.41],
            "r": [63107.39, 63118.22],
        }, "magnitudes": {
            "mag_image_1": {
                "i": [21.21, 20.42],
                "r": [20.87, 19.31],
            },
            "mag_image_2": {
                "i": [23.82, 22.87],
                "r": [23.45, 23.16],
            },
            "mag_image_3": {"i": [], "r": []},
            "mag_image_4": {"i": [], "r": []},
        }, "errors_low": {
            "mag_error_image_1_low": {
                "i": [0.04, 0.03],
                "r": [0.03, 0.02],
            },
            "mag_error_image_2_low": {
                "i": [0.06, 0.05],
                "r": [0.04, 0.03],
            },
        }, "errors_high": {
            "mag_error_image_1_high": {
                "i": [0.05, 0.04],
                "r": [0.03, 0.02],
            },
            "mag_error_image_2_high": {
                "i": [0.07, 0.06],
                "r": [0.05, 0.04],
            },
        },
        }
    :return: lightcurve plots.
    """
    magnitudes = lightcurve_dict["magnitudes"]
    errors_low = lightcurve_dict["errors_low"]
    errors_high = lightcurve_dict["errors_high"]
    obs_time = lightcurve_dict["obs_time"]
    # Extract all bands and filter out bands where all magnitudes are not NaN across.
    bands = [
        band
        for band in obs_time.keys()
        if any(
            not np.all(np.isnan(magnitudes[image_key][band]))
            for image_key in magnitudes.keys()
            if image_key.startswith("mag_image_")
        )
    ]

    # Identify non-empty magnitudes dynamically
    image_keys = []
    for key in magnitudes.keys():
        if key.startswith("mag_image_"):
            is_non_empty = any(
                not np.all(np.isnan(magnitudes[key][band])) for band in bands
            )
            if is_non_empty:
                image_keys.append(key)

    # Prepare the plot grid: rows for bands, columns for images +
    # optional images montage
    fig, axs = plt.subplots(
        nrows=len(bands),
        ncols=len(image_keys),
        figsize=(12, 6),
        gridspec_kw={"hspace": 0.6, "wspace": 0.3},
    )

    # Adjust axes for single-row scenarios
    if len(bands) == 1:
        axs = axs[np.newaxis, :]  # Ensure axs is 2D

    # Add titles for each column
    for col_idx, image_key in enumerate(image_keys):
        axs[0, col_idx].set_title(
            f"Lightcurves of image {col_idx+1}", fontsize=12, loc="center"
        )

    # Plot data for each band
    for row_idx, band in enumerate(bands):
        band_time = obs_time[band]

        for col_idx, image_key in enumerate(image_keys):
            mag_band = magnitudes[image_key][band]
            err_low_band = errors_low[
                f"{image_key.replace('mag_image', 'mag_error_image')}_low"
            ][band]
            err_high_band = errors_high[
                f"{image_key.replace('mag_image', 'mag_error_image')}_high"
            ][band]
            err_band = [err_low_band, err_high_band]

            # Plot the lightcurve for the current image
            axs[row_idx, col_idx].errorbar(
                band_time,
                mag_band,
                yerr=err_band,
                fmt=".",
                label=f"{band}-band",
                color=f"C{row_idx}",
                alpha=0.7,
            )
            axs[row_idx, col_idx].set_ylim(None, 30)
            axs[row_idx, col_idx].set_ylabel(f"Mag_{band}", fontsize=10)
            axs[row_idx, col_idx].invert_yaxis()
            axs[row_idx, col_idx].tick_params(axis="both", labelsize=8)

            # Add x-label only for the bottom row
            if row_idx == len(bands) - 1:
                axs[row_idx, col_idx].set_xlabel("MJD [Days]", fontsize=10)
    # Adjust layout to avoid overlaps
    plt.tight_layout()
    return fig


def create_montage(images_band, grid_size=None):
    """Creates a montage from a list of images, limited to the first 3 images,
    with consistent scaling. This function is a helper function for
    plot_lightcurves() function.

    :param images_band: List of 2D NumPy arrays representing images.
    :param grid_size: Tuple specifying the grid dimensions (rows, cols).
        If None, calculates the grid size to be approximately square.
    :return: 2D NumPy array representing the montage.
    """
    # Limit to the first 3 images
    images_band = images_band[:3]

    # Ensure all elements in images_band are 2D NumPy arrays
    images_band = [np.array(img) for img in images_band]

    # Determine the global minimum and maximum pixel values across all images
    global_min = min(np.min(img) for img in images_band)
    global_max = max(np.max(img) for img in images_band)

    # Normalize all images to the range [0, 1] based on global min and max
    normalized_images = [
        (img - global_min) / (global_max - global_min) for img in images_band
    ]

    # Determine grid size if not provided
    n_images = len(normalized_images)
    if grid_size is None:
        grid_cols = n_images
        grid_rows = 1
    else:
        grid_rows, grid_cols = grid_size

    # Determine the size of each image
    img_h, img_w = normalized_images[0].shape  # Assuming all images have the same shape

    # Create an empty array for the montage
    montage = np.zeros((grid_rows * img_h, grid_cols * img_w))

    # Fill the montage with images
    for idx, image in enumerate(normalized_images):
        row = idx // grid_cols
        col = idx % grid_cols
        montage[row * img_h : (row + 1) * img_h, col * img_w : (col + 1) * img_w] = (
            image
        )
    return montage


# microlensing lightcurve plot along with the magnification maps
def plot_lightcurves_and_magmap(
    convolved_map,
    lightcurves,
    time_duration_observer_frame,
    tracks,
    magmap_instance: MagnificationMap,
    lightcurve_type="magnitude",
):
    """Plot the lightcurves and the magnification map.

    :param convolved_map: convolved magnification map 2D numpy array.
        This is the map that is used to generate the lightcurves.
    :param lightcurves: list of lightcurves to plot
    :param time_duration_observer_frame: time duration in observer frame
        in days
    :param tracks: list of tracks to plot
    :param magmap_instance: instance of the MagnificationMap class. Must
        be the same as the one used to generate the lightcurves.
    :param lightcurve_type: type of lightcurve to plot. Can be
        'magnitude' or 'magnification'.
    :return: ax: the axis of the plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[2, 1])

    time_array = np.linspace(
        0, time_duration_observer_frame, len(lightcurves[0])
    )  # in days

    # light curves
    for i in range(len(lightcurves)):
        ax[0].plot(time_array, lightcurves[i], label=f"Lightcurve {i+1}")
    ax[0].set_xlabel("Time (days)")

    if lightcurve_type == "magnitude":
        ax[0].set_ylabel(
            "Magnitude $\\Delta m = -2.5 \\log_{10} (\\mu / \\mu_{\\text{av}})$"
        )
        im_to_show = -2.5 * np.log10(convolved_map / np.abs(magmap_instance.mu_ave))
    elif lightcurve_type == "magnification":
        ax[0].set_ylabel("Magnification $\\mu$")
        im_to_show = convolved_map

    ax[0].set_ylim(np.nanmin(im_to_show), np.nanmax(im_to_show))
    ax[0].legend()

    # magmap
    conts = ax[1].imshow(
        im_to_show,
        cmap="viridis_r",
        extent=[
            (magmap_instance.center_x - magmap_instance.half_length_x)
            / magmap_instance.theta_star,
            (magmap_instance.center_x + magmap_instance.half_length_x)
            / magmap_instance.theta_star,
            (magmap_instance.center_y - magmap_instance.half_length_y)
            / magmap_instance.theta_star,
            (magmap_instance.center_y + magmap_instance.half_length_y)
            / magmap_instance.theta_star,
        ],
        origin="lower",
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(conts, cax=cax)
    if lightcurve_type == "magnitude":
        cbar.set_label(
            "Microlensing $\\Delta m = -2.5 \\log_{10} (\\mu / \\mu_{\\text{av}})$ (magnitudes)"
        )
    elif lightcurve_type == "magnification":
        cbar.set_label("Microlensing magnification $\\mu$")
    ax[1].set_xlabel("$x / \\theta_★$")
    ax[1].set_ylabel("$y / \\theta_★$")
    # tracks are in pixel coordinates
    # to map them to the magmap coordinates, we need to convert them to the physical coordinates
    delta_x = 2 * magmap_instance.half_length_x / magmap_instance.num_pixels_x
    delta_y = 2 * magmap_instance.half_length_y / magmap_instance.num_pixels_y
    mid_x_pixel = magmap_instance.num_pixels_x // 2
    mid_y_pixel = magmap_instance.num_pixels_y // 2
    if tracks is not None:
        for j in range(len(tracks)):
            ax[1].plot(
                (tracks[j][1] - mid_x_pixel) * delta_x / magmap_instance.theta_star,
                (tracks[j][0] - mid_y_pixel) * delta_y / magmap_instance.theta_star,
                "w-",
                lw=1,
            )
            ax[1].text(
                (tracks[j][1][0] - mid_x_pixel) * delta_x / magmap_instance.theta_star,
                (tracks[j][0][0] - mid_y_pixel) * delta_y / magmap_instance.theta_star,
                str(j + 1),
                color="white",
                fontsize=16,
            )

    return ax


def plot_magnification_map(magmap_instance, ax=None, plot_magnitude=True, **kwargs):
    """Plot the magnification map on the given axis.

    :param magmap_instance: instance of the MagnificationMap class.
    :param ax: axis to plot on. If None, a new figure and axis will be
        created.
    :param plot_magnitude: if True, plot the magnitudes. If False, plot
        the magnifications.
    :param kwargs: additional keyword arguments to pass to the imshow
        function.
    :return: ax: the axis of the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if plot_magnitude:
        im = ax.imshow(
            magmap_instance.magnitudes,
            extent=[
                (magmap_instance.center_x - magmap_instance.half_length_x)
                / magmap_instance.theta_star,
                (magmap_instance.center_x + magmap_instance.half_length_x)
                / magmap_instance.theta_star,
                (magmap_instance.center_y - magmap_instance.half_length_y)
                / magmap_instance.theta_star,
                (magmap_instance.center_y + magmap_instance.half_length_y)
                / magmap_instance.theta_star,
            ],
            **kwargs,
        )
    else:
        im = ax.imshow(
            magmap_instance.magnifications,
            extent=[
                (magmap_instance.center_x - magmap_instance.half_length_x)
                / magmap_instance.theta_star,
                (magmap_instance.center_x + magmap_instance.half_length_x)
                / magmap_instance.theta_star,
                (magmap_instance.center_y - magmap_instance.half_length_y)
                / magmap_instance.theta_star,
                (magmap_instance.center_y + magmap_instance.half_length_y)
                / magmap_instance.theta_star,
            ],
            **kwargs,
        )
    ax.set_xlabel("$x / \\theta_★$")
    ax.set_ylabel("$y / \\theta_★$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    if plot_magnitude:
        cbar.set_label("Microlensing $\\Delta m$ (magnitudes)")
    else:
        cbar.set_label("Microlensing magnification")
