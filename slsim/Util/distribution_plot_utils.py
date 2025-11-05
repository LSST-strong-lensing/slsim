import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.patches import Patch

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
    if show_every_title:
        title_color = "white"
        title_fontsize = 1
    else:
        title_color = colors[-1]
        try:
            title_fontsize = CORNER_KWARGS["title_kwargs"]["fontsize"]
        except KeyError:
            title_fontsize = 30
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
                except KeyError:
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
        i += 1
        if show_every_title:
            titles_old = []
            for ax in fig.axes:
                # print(ax.get_title('left'))
                titles_old.append(ax.get_title("left"))
            titles_old = np.array(titles_old)
            titles_old = titles_old.reshape(
                i,
                len(titles_old) // i,
            ).T
            new_tit_i = 0

            for ax in fig.axes:
                curr_ax_title = ax.get_title("left")
                if curr_ax_title != "":
                    titles_curr = titles_old[new_tit_i]
                    color_i = 0
                    inch = 0
                    for tit in titles_curr:
                        ax.text(
                            0,
                            1.25 - inch,
                            tit,
                            color=colors[color_i],
                            weight=5,
                            fontsize=35,
                            transform=ax.transAxes,
                        )
                        inch += 0.1
                        color_i += 1
                    new_tit_i += i

    fig.legend(
        handles=legend_elements, frameon=False, ncol=1, loc=(0.58, 0.8), fontsize=30
    )
    # fig.tight_layout()
    if save_fig:
        if isinstance(save_fig, str):
            plt.savefig(save_fig, facecolor="white", bbox_inches="tight")
        else:
            plt.savefig("newfig.pdf", facecolor="white", bbox_inches="tight")
    return fig
