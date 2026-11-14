import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    label,
    maximum_filter,
)
from skimage.measure import find_contours
from skimage.morphology import thin

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.data_util import magnitude2amplitude
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util.util import rotate

from slsim.Lenses.lens import Lens

# the coarse pass looks this many times further out from the source than the images are traced so thin images arent missed
_COARSE_FACTOR = 2


def trace_finite_source_images(
    lens: Lens,
    search_window=40,
    pix_width_fine=0.1,
    x_center=0,
    y_center=0,
    cut=True,
    mag_per_arcsec2_cutoff=25,
    mag_band="F158",
    factor_coarse=12,
    factor_second=4,
    return_interior=False,
):
    """Finds the outline of every source's images in the search window based on
    an intensity cutoff of mag_per_arcsec2_cutoff in mag_band.

    First ray shoots a coarse grid to find roughly where the images of
    each source are. The first pass looks out to twice the radius the
    images are traced to so thin sources aren't missed. The coarse
    pixels it finds are divided into a smaller grid and ray shot again.
    Then starting from those points, each hit is expanded outwards until
    it finds the boundaries.

    By default filters out small singly-imaged sources, set cut=False to
    find all images for all sources

    :param lens: slsim.Lens
    :param search_window: width of the square search window
    :param pix_width_fine: width of a fine pixel
    :param x_center: center of the search window
    :param y_center: center of the search window
    :param cut: skip sources that have a single small image. Default
        True
    :param mag_per_arcsec2_cutoff: images are traced out to where the
        source is this faint, in mag per arcsec^2
    :param mag_band: band in which intensity is calculated. Defaults to
        Roman F158, must change if not generating Roman magnitudes
    :param factor_coarse: width of a coarse pixel, in fine pixels, must
        be a multiple of factor_second
    :param factor_second: width of a second pass pixel, in fine pixels
    :param return_interior: return the pixels inside each image instead
        of the outline
    :return: list of lists of (X array, Y array), in arcsecs. These are
        the traced outlines / interiors of every image for each source.
    """

    lens_model, kwargs_lens = lens.deflector_mass_model_lenstronomy(
        multi_plane=False, use_jax=False
    )
    sources = lens._source

    # pixel widths
    coarse_width = factor_coarse * pix_width_fine
    second_width = factor_second * pix_width_fine
    second_per_coarse = factor_coarse // factor_second

    # grid sizes
    num_pix = int(round(search_window / coarse_width))
    num_second = num_pix * second_per_coarse
    num_fine = num_pix * factor_coarse

    def pixel_centers(num, width, center):
        return center + (np.arange(num) - (num - 1) / 2) * width

    coarse_x = pixel_centers(num_pix, coarse_width, x_center)
    coarse_y = pixel_centers(num_pix, coarse_width, y_center)
    second_x = pixel_centers(num_second, second_width, x_center)
    second_y = pixel_centers(num_second, second_width, y_center)
    fine_x = pixel_centers(num_fine, pix_width_fine, x_center)
    fine_y = pixel_centers(num_fine, pix_width_fine, y_center)

    # the index of the coarse pixel that each second pass pixel is inside of
    coarse_index_of_second = np.arange(num_second) // second_per_coarse

    # per source
    alpha_scalings = lens_model._lensCosmo.beta_double_source_plane(
        z_lens=np.atleast_1d(lens.deflector_redshift)[0],  # can be an array
        z_source_1=[s.redshift for s in sources],
        z_source_2=lens_model._z_source_convention,
    )
    alpha_scaling_ratios = (
        np.asarray(alpha_scalings) / lens_model.lens_model._alpha_scaling
    )

    # trace out to this many half light radii, per source
    n_sersic, _, amp = _sersic_params(
        [s.kwargs_extended_light(mag_band) for s in sources], mag_per_arcsec2_cutoff
    )
    max_radii = solve_max_rad_re(n_sersic, amp, 1)

    # the coarse grid is the same for every source
    x_coarse, y_coarse = np.meshgrid(coarse_x, coarse_y, indexing="ij")
    coarse_alpha_x, coarse_alpha_y = lens_model.alpha(x_coarse, y_coarse, kwargs_lens)

    # keep the alphas already computed and only call the lens model for new pixels
    second_alpha_x = np.zeros((num_second, num_second))
    second_alpha_y = np.zeros((num_second, num_second))
    second_computed = np.zeros((num_second, num_second), dtype=bool)
    fine_alpha_x = np.zeros((num_fine, num_fine))
    fine_alpha_y = np.zeros((num_fine, num_fine))
    fine_computed = np.zeros((num_fine, num_fine), dtype=bool)

    # per source, keyed by source index
    #   inside: fine pixels known to be in an image
    #   checked: fine pixels already ray shot (or in the seed)
    #   todo: fine pixels to ray shoot next
    inside = {}
    checked = {}
    todo = {}

    for i, source in enumerate(sources):
        # find coarse pixels that land near the source
        distance = dist_hlr(
            source,
            x_coarse - coarse_alpha_x * alpha_scaling_ratios[i],
            y_coarse - coarse_alpha_y * alpha_scaling_ratios[i],
        )
        near = distance <= _COARSE_FACTOR * max_radii[i]

        if not near.any():
            continue

        # second pass, a second pass pixel is near if its coarse pixel is near
        near_second = near[coarse_index_of_second][:, coarse_index_of_second]
        # pixels that need to be ray shot
        needed = near_second & ~second_computed

        if needed.any():
            mx, my = np.nonzero(needed)
            second_alpha_x[mx, my], second_alpha_y[mx, my] = lens_model.alpha(
                second_x[mx], second_y[my], kwargs_lens
            )
            second_computed |= needed

        sx, sy = np.nonzero(near_second)
        x = second_x[sx] - alpha_scaling_ratios[i] * second_alpha_x[sx, sy]
        y = second_y[sy] - alpha_scaling_ratios[i] * second_alpha_y[sx, sy]
        is_hit = dist_hlr(source, x, y) <= max_radii[i]
        sx, sy = sx[is_hit], sy[is_hit]

        if len(sx) == 0:
            continue

        hit = np.zeros((num_pix, num_pix), dtype=bool)
        hit[sx // second_per_coarse, sy // second_per_coarse] = True

        if cut:
            # skip if small single image
            images, num_images = label(
                binary_fill_holes(hit), structure=np.ones((3, 3))
            )
            xs, ys = np.nonzero(images)

            if (
                num_images == 1
                and np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) <= 10
            ):
                continue

        # fine pass, first fill in the middle of each image without ray
        # shooting it. distance is only known on the coarse grid, so
        # interpolate it onto the fine pixels, then erode a couple of pixels
        # so the seed stays safely inside. the growing below only goes
        # outwards, so a seed that's too big would never stop growing
        spline = RectBivariateSpline(
            coarse_x, coarse_y, distance - max_radii[i], kx=3, ky=3
        )
        labels, num_images = label(hit, structure=np.ones((3, 3)))
        inside[i] = np.zeros((num_fine, num_fine), dtype=bool)

        for n in range(1, num_images + 1):
            image_x, image_y = np.nonzero(labels == n)

            if len(image_x) < 3:
                continue

            # fine pixels covering the image's coarse pixels plus a margin
            # of one coarse pixel (+ 1 on the stop to include the edge)
            x_start = max(image_x.min() - 1, 0) * factor_coarse
            x_stop = min(image_x.max() + 1, num_pix) * factor_coarse + 1
            y_start = max(image_y.min() - 1, 0) * factor_coarse
            y_stop = min(image_y.max() + 1, num_pix) * factor_coarse + 1

            seed = binary_erosion(
                spline(fine_x[x_start:x_stop], fine_y[y_start:y_stop]) < 0,
                iterations=2,
            )
            inside[i][x_start:x_stop, y_start:y_stop] |= seed

        checked[i] = inside[i].copy()

        # also start from the fine pixel in the middle of each second pass
        # hit, this is what starts the small and thin images (no seed)
        middle = factor_second // 2
        start = np.zeros((num_fine, num_fine), dtype=bool)
        start[sx * factor_second + middle, sy * factor_second + middle] = True

        todo[i] = binary_dilation(start | inside[i]) & ~checked[i]

    # grow all the images together, so each step is one lens model call
    todo = {i: mask for i, mask in todo.items() if mask.any()}

    while todo:
        # ray shoot the pixels that any source wants and nobody has done yet
        needed = np.any(list(todo.values()), axis=0) & ~fine_computed

        if needed.any():
            mx, my = np.nonzero(needed)
            fine_alpha_x[mx, my], fine_alpha_y[mx, my] = lens_model.alpha(
                fine_x[mx], fine_y[my], kwargs_lens
            )
            fine_computed |= needed

        for i, mask in todo.items():
            # which of this source's pixels land inside the source
            px, py = np.nonzero(mask)
            x = fine_x[px] - alpha_scaling_ratios[i] * fine_alpha_x[px, py]
            y = fine_y[py] - alpha_scaling_ratios[i] * fine_alpha_y[px, py]
            is_inside = dist_hlr(sources[i], x, y) < max_radii[i]

            new = np.zeros((num_fine, num_fine), dtype=bool)
            new[px[is_inside], py[is_inside]] = True

            checked[i] |= mask
            inside[i] |= new

            # only the neighbors of new inside pixels can be inside next
            todo[i] = binary_dilation(new) & ~checked[i]

        # sources with nothing left to check are done
        todo = {i: mask for i, mask in todo.items() if mask.any()}

    # turn the inside masks into outlines (or interiors)
    all_images = [[] for _ in sources]

    for i, mask in inside.items():
        if np.count_nonzero(mask) < 5:  # drop tiny images
            continue

        mask = binary_fill_holes(mask)

        if return_interior:
            labels, num_images = label(mask, structure=np.ones((3, 3)))

            for n in range(1, num_images + 1):
                xs, ys = np.nonzero(labels == n)
                all_images[i].append((fine_x[xs], fine_y[ys]))
        else:
            # pad so contours close at the edge. the mask is transposed, so
            # contour rows are y and columns are x, both shifted by the padding
            padded = np.pad(mask.T, 1).astype(float)

            for contour in find_contours(padded, 0.5, fully_connected="high"):
                rows, cols = contour[:, 0], contour[:, 1]
                x = fine_x[0] + (cols - 1) * pix_width_fine
                y = fine_y[0] + (rows - 1) * pix_width_fine
                all_images[i].append((x, y))

    return all_images


def _sersic_params(sources, mag_zero_point):
    """Returns (n_sersic, r_eff, amp) arrays given light dicts of sersic
    sources.

    :param sources: list of (light_model_list, kwargs_light_mag) tuples
    :param mag_zero_point: scalar, or array-like with one zero point per
        source
    :return: n_sersic, r_eff (arcsec), amp arrays
    """

    n_sersic = np.empty(len(sources))
    r_eff = np.empty(len(sources))
    amp = np.empty(len(sources))

    for i, (model_list, kwargs_mag) in enumerate(sources):
        light_model = LightModel(list(model_list))

        kwargs_amp = magnitude2amplitude(light_model, kwargs_mag, mag_zero_point)
        kw = kwargs_amp[0]
        n_sersic[i] = kw["n_sersic"]
        r_eff[i] = kw["R_sersic"]
        amp[i] = kw["amp"]

    return n_sersic, r_eff, amp


def solve_max_rad_re(n_sersic, amp, amp_min):
    """Given a sersic number and amplitude at the half light radius, solve for
    the radius (in terms of half light radius) where amp is amp_min.

    :param n_sersic: sersic number
    :param amp: amplitude at half light radius
    :param amp_min: amplitude to solve for
    """

    bn = 1.9992 * n_sersic - 0.3271

    ratio = 1 + np.log(amp / amp_min) / bn
    ratio = np.clip(ratio, 0, None)

    return ratio**n_sersic


def dist_hlr(source, x, y):
    """Returns the distance from (x, y) to the source accounting for
    ellipticity, in terms of the source's half light radius. Proportional to
    the source's surface brightness at that point.

    :param source: slsim.Source
    :param x: x source plane position, arcsecs
    :param y: y source plane position, arcsecs
    """

    phi, q = ellipticity2phi_q(*source.ellipticity)
    phi = np.pi / 2 + phi

    x = x - source.extended_source_position[0]
    y = y - source.extended_source_position[1]

    x, y = rotate(x, y, -phi)

    x = x * np.sqrt(q)
    y = y / np.sqrt(q)

    return np.hypot(x, y) / source.angular_size


def _find_midline(image, pix_width=0.1):
    """
    image: (X, Y) interior pixels, not boundary
    Returns (x, y, dist_edge) midline points arrays
    """

    X, Y = image

    # Convert physical coordinates to grid indices
    ix = np.floor(X / pix_width).astype(int)
    iy = np.floor(Y / pix_width).astype(int)

    # Crop to image
    ix0 = ix.min()
    iy0 = iy.min()

    ix_local = ix - ix0
    iy_local = iy - iy0

    nx = ix_local.max() + 1
    ny = iy_local.max() + 1

    mask = np.zeros((nx, ny), dtype=bool)
    mask[ix_local, iy_local] = True

    # Distance from each point to the boundary
    distance = distance_transform_edt(mask)

    # Local maximum of the distance transform
    local_max = maximum_filter(distance, size=9)

    # Keep points close to the local maximum
    ridge = (distance >= local_max - 2.0) & mask

    # Lightly close small gaps
    if min(ridge.shape) >= 5:
        ridge = binary_closing(
            ridge, structure=np.ones((5, 5), dtype=bool), iterations=3
        )

    ridge = thin(ridge)

    ridge &= distance > 1.0

    # Output points
    sx, sy = np.nonzero(ridge)

    X_center = (sx + ix0) * pix_width
    Y_center = (sy + iy0) * pix_width

    return (X_center, Y_center, distance[ridge] * pix_width)


def compute_LW_phi(arc, pix_width=0.1):
    """
    Calculates the length, width, and orientation angle of an arc
    Length: finds midpoint, farthest point from midpoint, and farthest point from that point. Length is distance between the two last points.
    Width: finds geometric midline, width is twice the average distance to the edge on that midline
    Orientation: computes dot product of average direction vector with center position vector. Value of 0 indicates tangential arc, 1 indicates radial arc.

    :param arc: (X, Y) of X and Y coordinate arrays, interior pixels, not boundary.
    :param pix_width: fine pixel width used in trace_finite_source_images

    :returns: (length, width, orientation)
    """

    _, _, distedge = _find_midline(arc, pix_width)
    W = 2 * np.mean(distedge)

    X, Y = arc

    cx, cy = np.mean(X), np.mean(Y)

    p1_idx = np.argmax(np.hypot(X - cx, Y - cy))
    p1x, p1y = X[p1_idx], Y[p1_idx]

    p2_idx = np.argmax(np.hypot(X - p1x, Y - p1y))
    p2x, p2y = X[p2_idx], Y[p2_idx]

    ang_vector = (p2x - p1x, p2y - p1y)
    L = np.hypot(*ang_vector)

    phi = np.abs(
        np.dot(ang_vector, (cx, cy)) / np.hypot(cx, cy) / np.hypot(*ang_vector)
    )

    return L, W, phi
