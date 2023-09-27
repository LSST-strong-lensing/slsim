import matplotlib.pyplot as plt
import random

"""This module contains various plotting definations."""


def create_image_montage_from_image_list(num_rows, num_cols, images, time=None):
    """Creates an image montage from an image list.

    :param num_rows: number of images to display horizontally
    :param num_cols: number of images to display vertically
    :param images: list of images
    :return: image montage of given images.
    """

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    for i in range(num_rows):
        for j in range(num_cols):
            if i * num_cols + j < len(images):
                image = images[i * num_cols + j]

                if time is None:
                    axes[i, j].imshow(image, origin="lower")
                    axes[i, j].axis("off")  # Turn off axis labels
                else:
                    axes[i, j].imshow(image, origin="lower")
                    axes[i, j].text(
                        0.05,
                        0.95,
                        f"Time: {round(time[i * num_cols + j],2)}",
                        fontsize=10,
                        verticalalignment="top",
                        horizontalalignment="left",
                        transform=axes[i, j].transAxes,
                    )
                    axes[i, j].axis("off")  # Turn off axis labels

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    return None


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
    return None
