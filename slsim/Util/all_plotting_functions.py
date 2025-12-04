import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import corner
from matplotlib.patches import Patch

from tqdm import tqdm
from PIL import Image
import matplotlib.image as mpimg
from astropy.visualization import (
    ImageNormalize,
    SqrtStretch,
    LinearStretch,
    LogStretch,
    MinMaxInterval,
    ZScaleInterval,
)


def plot_collage(
    image_folder: str,
    N_rows=2,
    N_cols=2,
    vmin=None,
    vmax=None,
    stretch=None,
    interval=None,
    figsize=None,
    tight_layout=False,
    title=None,
    fontsize=18,
    h5_filename=None,
    cmap=None,
    return_fig=False,
    ax=None,
    seed=1,
):
    assert interval in [None, "zscale", "minmax"]
    assert stretch in [None, "sqrt", "linear", "log"]
    assert (interval is None and stretch is None) or (
        interval in ["zscale", "minmax"] and stretch in ["sqrt", "linear", "log"]
    )
    if stretch is None or interval is None:
        print(f"Will use vmin/vmax of {(vmin,vmax)} and ignore interval/stretch values")
    else:
        print(
            f"Will use interval/stretch of {(interval,stretch)} and vmin/vmax of {(vmin,vmax)}"
        )
    np.random.seed(seed)
    N_images = int(N_rows * N_cols)
    image_npy_list = glob.glob(f"{image_folder}/*.npy")
    image_h5_list = glob.glob(f"{image_folder}/*.h5")
    image_jpeg_list = glob.glob(f"{image_folder}/*.jpeg")
    image_png_list = glob.glob(f"{image_folder}/*.png")
    if len(image_npy_list) > 0:
        print("Plotting from npy files")
        random_indx = np.random.choice(
            np.arange(len(image_npy_list)), replace=False, size=N_images
        )
        image_list = [
            np.load(image_npy_list[random_indx[n_im]]) for n_im in range(N_images)
        ]
    elif len(image_h5_list) > 0:
        print("Plotting from h5 files")
        if h5_filename is None:
            assert (
                len(image_h5_list) == 1
            )  # There should only be one h5 file in the folder
        else:
            image_h5_list = [h5_filename]
        with h5py.File(image_h5_list[0], "r") as f0:
            number_of_files = f0["data"].shape[0]
            h5_file_array = f0["data"][()]
            random_indx = np.random.choice(
                np.arange((number_of_files)), replace=False, size=N_images
            )
            image_list = [
                h5_file_array[random_indx[n_im]] for n_im in tqdm(range(N_images))
            ]
    elif len(image_jpeg_list) > 0:
        print("Plotting from jpeg files")
        random_indx = np.random.choice(
            np.arange(len(image_jpeg_list)), replace=False, size=N_images
        )
        print(image_jpeg_list[random_indx[0]])
        image_list = [
            mpimg.imread(image_jpeg_list[random_indx[n_im]])
            for n_im in tqdm(range(N_images))
        ]
    elif len(image_png_list) > 0:
        print("Plotting from png files")
        random_indx = np.random.choice(
            np.arange(len(image_png_list)), replace=False, size=N_images
        )
        print(image_png_list[random_indx[0]])
        image_list = [
            np.array(Image.open(image_png_list[random_indx[n_im]]))
            for n_im in tqdm(range(N_images))
        ]

    if figsize is None:
        figsize = (N_cols, N_rows)
    if not return_fig:
        fig, ax = plt.subplots(N_rows, N_cols, figsize=figsize)
    for n_im in range(N_images):
        x = n_im % N_rows
        y = np.floor(n_im / N_rows).astype("int")
        if stretch is None or interval is None:
            ax[x, y].imshow(image_list[n_im], vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            norm_dict = {
                "minmax": MinMaxInterval,
                "zscale": ZScaleInterval,
                "sqrt": SqrtStretch,
                "linear": LinearStretch,
                "log": LogStretch,
            }
            norm = ImageNormalize(
                image_list[n_im],
                interval=norm_dict[interval](),
                stretch=norm_dict[stretch](),
                vmin=vmin,
                vmax=vmax,
            )
            ax[x, y].imshow(image_list[n_im], norm=norm, cmap=cmap)
        # Remove axis ticks:
        ax[x, y].xaxis.set_tick_params(labelbottom=False)
        ax[x, y].yaxis.set_tick_params(labelleft=False)
        # Hide X and Y axes tick marks
        ax[x, y].set_xticks([])
        ax[x, y].set_yticks([])
    if title is not None:
        plt.suptitle(title, fontsize=fontsize, fontweight="bold")
    if tight_layout:
        plt.tight_layout()
    if return_fig:
        return ax
    plt.show()


CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=30, loc="left"),
    plot_density=True,
    plot_datapoints=True,
    fill_contours=True,
    show_titles=True,
    max_n_ticks=10,
    bins=20,
)


def get_range(dist):
    """Gets n-range for n-dist.

    Args:
        dist (np.ndarray): distribution of n parameters

    Returns:
        list[tuple]: list of n tuples containing distribution range of n parameters
    """
    range_arr = np.array([np.min(dist, axis=0), np.max(dist, axis=0)]).T
    final = [tuple(i) for i in range_arr]
    return final


def get_mean(dist):
    """Get n-mean from n-dist.

    Args:
        dist (n-d array): array containing predictions/truths etc

    Returns:
        array: array of means along axis 0.
    """
    return np.mean(dist, axis=0)


def make_contour(
    list_of_dists,
    labels,
    categories,
    colors,
    range_for_bin=False,
    show_correlation=False,
    truths_list=None,
    show_every_title=False,
    save_fig=False,
):
    cat_to_col = dict(zip(categories, colors))
    legend_elements = []
    for cat in categories:
        legend_elements.append(
            Patch(facecolor=cat_to_col[cat], edgecolor=cat_to_col[cat], label=cat)
        )

    exemplar_dist = list_of_dists[0]
    if range_for_bin:
        bin_range = get_range(exemplar_dist)
    else:
        bin_range = None

    fig, ax = plt.subplots(
        exemplar_dist.shape[1], exemplar_dist.shape[1], figsize=(20, 22)
    )

    title_color = colors[-1]
    try:
        title_fontsize = CORNER_KWARGS["title_kwargs"]["fontsize"]
    except KeyError:
        title_fontsize = 50
    try:
        CORNER_KWARGS["title_kwargs"].update(color=title_color, fontsize=title_fontsize)
    except KeyError:
        CORNER_KWARGS["title_kwargs"] = dict(fontsize=title_fontsize, color=title_color)
    i = 0
    alpha = 0.3
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=15)

    for dist in list_of_dists:
        means = get_mean(dist)
        if truths_list is None:
            truths = means
            truth_color = colors[i]
        else:
            truths = truths_list[i]
            truth_color = "red"

        corner.corner(
            data=dist,
            labels=labels,
            color=colors[i],
            truths=truths,
            hist_kwargs=dict(density=True, lw=5, color=colors[i], range=bin_range),
            levels=[0.68, 0.95],
            truth_color=truth_color,
            **CORNER_KWARGS,
            title_fmt=".2f",
            fig=fig,
            alpha=alpha,
        )
        corner.overplot_lines(fig, means, color=colors[i])
        if truths_list is not None:
            corner.overplot_lines(fig, truths, color="black", linewidth=2)
        alpha = alpha + len(list_of_dists) / 10
        alpha = max(1, alpha)
        props1 = dict(boxstyle="round", facecolor="white")
        if show_correlation:
            mini_fig = corner.corner(data=dist)
            plt.figure(visible=False)
            to_put = {}
            ax_i = 0
            for ax in mini_fig.get_axes():
                line = ax.lines
                try:
                    # print(line[0].get_xdata(), line[0].get_ydata())
                    r_coef = np.corrcoef(line[0].get_xdata(), line[0].get_ydata())[0][1]
                    # print(r_coef)
                    to_put[ax_i] = np.round(r_coef, 2)
                except IndexError:
                    pass
                ax_i += 1
            fig_ax_list = fig.get_axes()
            for ax_stored in to_put.keys():
                ax = fig_ax_list[ax_stored]
                ax.text(
                    0.6,
                    0.9 - i / 10,
                    to_put[ax_stored],
                    size=18,
                    color=colors[i],
                    transform=ax.transAxes,
                    bbox=props1,
                )

        if show_every_title and i < len(list_of_dists) - 1:
            color_i = colors[i]
            inch = len(list_of_dists) * 0.06 * i
            for panel in range(len(fig.axes)):
                ax = fig.axes[panel]
                titles_curr = ax.get_title("left")
                if titles_curr == "":
                    continue
                ax.text(
                    0,
                    1.25 + (0.05 * len(list_of_dists)) - inch,
                    titles_curr,
                    color=color_i,
                    weight=5,
                    fontsize=title_fontsize,
                    transform=ax.transAxes,
                )
        i += 1

    # print(np.array(fig.get_axes()).shape)

    fig.legend(
        handles=legend_elements, frameon=False, ncol=1, loc=(0.58, 0.8), fontsize=30
    )
    # fig.tight_layout()
    if save_fig:
        if type(save_fig).isinstance(str):
            plt.savefig(save_fig, facecolor="white", bbox_inches="tight")
        else:
            plt.savefig("newfig.pdf", facecolor="white", bbox_inches="tight")
    return fig
