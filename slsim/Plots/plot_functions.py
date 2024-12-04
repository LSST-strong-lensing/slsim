import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import random
import numpy as np

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

def plot_lightcurves(data, images=True):
    """
    Plots lightcurves dynamically for all available images across different bands, 
     excluding empty images.
    :param data: Dictionary returned by `extract_lightcurves_in_different_bands`.
    :param images: Boolean. If True, plots some sample images in each bands.
    :return: lightcurve plots (and image motage).
    """
    magnitudes = data["magnitudes"]
    errors_low = data["errors_low"]
    errors_high = data["errors_high"]
    obs_time = data["obs_time"]
    if images:
        image_lists = data["image_lists"]

    # Extract all bands and filter out bands where all magnitudes are not NaN across.
    bands = [
        band for band in obs_time.keys()
        if any(
            not np.all(np.isnan(magnitudes[image_key][band]))
            for image_key in magnitudes.keys() if image_key.startswith("mag_image_")
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
        ncols=len(image_keys) + (1 if images else 0),
        figsize=(12, 6),
        gridspec_kw={"hspace": 0.6, "wspace": 0.3},
    )

    # Adjust axes for single-row or single-column scenarios
    if len(bands) == 1:
        axs = axs[np.newaxis, :]  # Ensure axs is 2D
    if len(image_keys) == 1 and images:
        axs = axs[:, np.newaxis]

    # Add titles for each column
    for col_idx, image_key in enumerate(image_keys):
        axs[0, col_idx].set_title(f"Lightcurves of image {col_idx+1}",
                                   fontsize=12, loc="center")
    if images:
        axs[0, -1].set_title("Lens Images", fontsize=12, loc="center")

    # Plot data for each band
    for row_idx, band in enumerate(bands):
        band_time = obs_time[band]

        for col_idx, image_key in enumerate(image_keys):
            mag_band = magnitudes[image_key][band]
            err_low_band = errors_low[
                f"{image_key.replace('mag_image', 'mag_error_image')}_low"][band]
            err_high_band = errors_high[
                f"{image_key.replace('mag_image', 'mag_error_image')}_high"][band]
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
            axs[row_idx, col_idx].set_ylabel(f"Mag_{band}", fontsize=10)
            axs[row_idx, col_idx].invert_yaxis() 
            axs[row_idx, col_idx].tick_params(axis="both", labelsize=8)

            # Add x-label only for the bottom row
            if row_idx == len(bands) - 1:
                axs[row_idx, col_idx].set_xlabel("MJD [Days]", fontsize=10)

        # Display lens images montage if image_lists is available
        if images:
            montage = create_montage(image_lists[band]) if image_lists[
                band] else np.zeros((10, 10))
            axs[row_idx, -1].imshow(montage, cmap="viridis", origin="lower")
            axs[row_idx, -1].axis("off")

    # Adjust layout to avoid overlaps
    plt.tight_layout()
    plt.show()
    return fig


def create_montage(images_band, grid_size=None):
    """
    Creates a montage from a list of images, limited to the first 3 images, with 
    consistent scaling. This function is a helper function for plot_lightcurves() 
    function.

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
    normalized_images = [(img - global_min) / (
        global_max - global_min) for img in images_band]

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
        montage[row * img_h : (row + 1) * img_h, col * img_w : (
            col + 1) * img_w] = image
    return montage
