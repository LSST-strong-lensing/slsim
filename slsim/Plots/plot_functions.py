import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import random
import numpy as np

"""This module contains various plotting definations."""


def create_image_montage_from_image_list(
    num_rows, num_cols, images, time=None, image_type="other", image_center=None
):
    """Creates an image montage from an image list.

    :param num_rows: number of images to display horizontally
    :param num_cols: number of images to display vertically
    :param images: list of images
    :param time: array of observation time for point source images. If None, considers
        static case.
    :param image_type: type of the provided image. It could be 'dp0' or any other name.
    :param image_center: center of the source images.
    :type image_center: array. eg: for two image, it should be like
        np.array([[13.71649063, 13.09556121], [16.69249276, 17.78106655]])
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
    """Creates an image montage of random lenses from the catalog of injected lens.

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
