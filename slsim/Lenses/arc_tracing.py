import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt, label,
                           maximum_filter)
from skimage.measure import find_contours
from skimage.morphology import thin

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.data_util import magnitude2amplitude
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util.util import rotate

from slsim.Lenses.lens import Lens

# the coarse pass looks this many times further out from the source than the images are traced so thin images arent missed
_COARSE_FACTOR = 2

def sersic_params(sources, mag_zero_point):
    """Returns (n_sersic, r_eff, amp) arrays given light dicts of sersic sources

    :param sources: list of (light_model_list, kwargs_light_mag) tuples
    :param mag_zero_point: scalar, or array-like with one zero point per source
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
    """
    Given a sersic number and amplitude at the half light radius, solve for the radius (in terms of half light radius) where amp is amp_min

    :param n_sersic: sersic number
    :param amp: amplitude at half light radius
    :param amp_min: amplitude to solve for
    """

    bn = 1.9992 * n_sersic - 0.3271

    ratio = 1 + np.log(amp / amp_min) / bn
    ratio = np.clip(ratio, 0, None)

    return ratio ** n_sersic


def dist_hlr(source, x, y):
    """Returns the distance from (x, y) to the source accounting for ellipticity, in terms of the source's half light radius.
    Proportional to the source's surface brightness at that point.

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


def trace_finite_source_images(lens: Lens, search_window=40, pix_width_fine=0.1, factor_coarse=12, factor_second=4, x_center=0, y_center=0,
                               cut=False, return_interior=False, mag_per_arcsec2_cutoff=25, mag_band="F158"):
    """Finds the outline of every source's images in the search window

    Ray shoots a coarse grid to find roughly where the images of each source are. The first pass looks out to twice the radius the images are traced to to avoid missing thin sources. The coarse pixels it finds are divided into a smaller grid and ray shot again. Starting from those points, each hit is expanded outwards until it finds the boundaries.

    This function was partially written by AI

    :param lens: slsim.Lens
    :param search_window: width of the square search window, arcsecs. Rounded to a whole number of coarse pixels
    :param pix_width_fine: width of a fine pixel
    :param factor_coarse: width of a coarse pixel, in fine pixels, must be a multiple of factor_second
    :param factor_second: width of a second pass pixel, in fine pixels
    :param x_center: center of the search window
    :param y_center: center of the search window
    :param cut: skip sources that have a single small image
    :param return_interior: return the pixels inside each image instead of its outline
    :param mag_per_arcsec2_cutoff: images are traced out to where the source is this faint, in mag per arcsec^2
    :param mag_band: band in which intensity is calculated. Defaults to Roman F158, must change if not generating Roman magnitudes

    Returns a list of lists of (X array, Y array), in arcsecs. These are the traced outlines / interiors of every image for each source.
    """

    lens_model, kwargs_lens = lens.deflector_mass_model_lenstronomy(multi_plane=False, use_jax=False)
    sources = lens._source

    # Pixel widths
    fine_width = pix_width_fine
    coarse_width = factor_coarse * fine_width
    second_width = factor_second * fine_width
    second_per_coarse = factor_coarse // factor_second

    # Number of pixels along one side of each grid, and the coordinates of the pixel centers
    num_pix = int(round(search_window / coarse_width))
    num_second = num_pix * second_per_coarse
    num_fine = num_pix * factor_coarse

    def pixel_centers(num, width, center):
        return center + (np.arange(num) - (num - 1) / 2) * width

    coarse_x = pixel_centers(num_pix, coarse_width, x_center)
    coarse_y = pixel_centers(num_pix, coarse_width, y_center)
    second_x = pixel_centers(num_second, second_width, x_center)
    second_y = pixel_centers(num_second, second_width, y_center)
    fine_x = pixel_centers(num_fine, fine_width, x_center)
    fine_y = pixel_centers(num_fine, fine_width, y_center)

    # alpha is computed at the lens model's own scaling, ratios rescale it for each source
    alpha_scalings = lens_model._lensCosmo.beta_double_source_plane(
        z_lens=np.atleast_1d(lens.deflector_redshift)[0],  # if deflector redshift is an array
        z_source_1=[s.redshift for s in sources],
        z_source_2=lens_model._z_source_convention,
    )
    ratios = np.asarray(alpha_scalings) / lens_model.lens_model._alpha_scaling

    # how far out to trace each source, in half light radii
    n_sersic, _, amp = sersic_params([s.kwargs_extended_light(mag_band) for s in sources], mag_per_arcsec2_cutoff)
    max_radii = solve_max_rad_re(n_sersic, amp, 1)

    # First pass, out to _COARSE_FACTOR times the radius. Keep the sources that have coarse pixels near enough
    x_coarse, y_coarse = np.meshgrid(coarse_x, coarse_y, indexing="ij")
    coarse_alpha_x, coarse_alpha_y = lens_model.alpha(x_coarse, y_coarse, kwargs_lens)

    candidates = []  # (source index, distance at the coarse pixels, second pass pixels)
    second_needed = np.zeros((num_second, num_second), dtype=bool)  # the second pass pixels of all the sources

    for i, source in enumerate(sources):
        # the distance to the source of every coarse pixel, after ray shooting it to the source plane
        distance = dist_hlr(source, x_coarse - coarse_alpha_x * ratios[i], y_coarse - coarse_alpha_y * ratios[i])
        near = distance <= _COARSE_FACTOR * max_radii[i]

        if near.any():
            # divide every near coarse pixel into second pass pixels
            sx, sy = np.nonzero(np.kron(near, np.ones((second_per_coarse, second_per_coarse), dtype=bool)))
            second_needed[sx, sy] = True
            candidates.append((i, distance, sx, sy))

    # Second pass, out to the true radius, so these are the hits
    second_alpha_x = np.zeros((num_second, num_second))
    second_alpha_y = np.zeros((num_second, num_second))

    if candidates:
        mx, my = np.nonzero(second_needed)
        second_alpha_x[mx, my], second_alpha_y[mx, my] = lens_model.alpha(second_x[mx], second_y[my], kwargs_lens)

    traced = []  # sources that get a fine pass: (source index, distance at the coarse pixels, coarse pixels with a hit, second pass pixels with a hit)

    for i, distance, sx, sy in candidates:
        x = second_x[sx] - ratios[i] * second_alpha_x[sx, sy]
        y = second_y[sy] - ratios[i] * second_alpha_y[sx, sy]
        is_hit = dist_hlr(sources[i], x, y) <= max_radii[i]
        sx, sy = sx[is_hit], sy[is_hit]

        # nothing hit, so there are no images to trace
        if len(sx) == 0:
            continue

        hit = np.zeros((num_pix, num_pix), dtype=bool)
        hit[sx // second_per_coarse, sy // second_per_coarse] = True

        if cut:
            # a single image (holes in it, like an Einstein ring, do not count as separate images) that is no longer than 5 coarse pixels along the
            # diagonal of its bounding box is skipped
            images, num_images = label(binary_fill_holes(hit), structure=np.ones((3, 3)))
            xs, ys = np.nonzero(images)

            if num_images == 1 and np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) <= 5:
                continue

        traced.append((i, distance, hit, sx, sy))

    # Fine pass. For every traced source: inside is the fine pixels known to be inside an image, known is the fine pixels that are inside a seed or
    # have been ray shot, and todo is the fine pixels that are ray shot next
    inside, known, todo = {}, {}, {}
    degree = min(3, num_pix - 1)

    for i, distance, hit, sx, sy in traced:
        inside[i] = np.zeros((num_fine, num_fine), dtype=bool)

        # The seed is the part of an image that is trusted to be inside without ray shooting it. The distance to the source is only known at the
        # coarse pixels, so it is interpolated onto the fine pixels around the image. The seed is where that says inside, but with a margin of a
        # couple of pixels. The ray shooting after this only goes outwards, so a seed that is too big would never get corrected.
        spline = RectBivariateSpline(coarse_x, coarse_y, distance - max_radii[i], kx=degree, ky=degree)
        labels, num_images = label(hit, structure=np.ones((3, 3)))

        for n in range(1, num_images + 1):
            xs, ys = np.nonzero(labels == n)

            # A small image is smaller than the coarse pixels can resolve, so the interpolation across it is not reliable
            if len(xs) < 3:
                continue

            # The coarse pixels around the image, with a margin of one
            x0, x1 = max(xs.min() - 1, 0) * factor_coarse, min(xs.max() + 2, num_pix) * factor_coarse
            y0, y1 = max(ys.min() - 1, 0) * factor_coarse, min(ys.max() + 2, num_pix) * factor_coarse

            seed = binary_erosion(spline(fine_x[x0:x1], fine_y[y0:y1]) < 0, iterations=2)
            inside[i][x0:x1, y0:y1] |= seed

        known[i] = inside[i].copy()

        # The second pass found points that are inside the images, ray shoot those and the pixels next to them as well. This also starts the small
        # and thin images, which do not have a seed
        start = np.zeros((num_fine, num_fine), dtype=bool)
        start_x = np.rint((second_x[sx] - fine_x[0]) / fine_width).astype(int)
        start_y = np.rint((second_y[sy] - fine_y[0]) / fine_width).astype(int)
        start[np.clip(start_x, 0, num_fine - 1), np.clip(start_y, 0, num_fine - 1)] = True

        todo[i] = binary_dilation(start | inside[i]) & ~known[i]

    fine_alpha_x = np.zeros((num_fine, num_fine))
    fine_alpha_y = np.zeros((num_fine, num_fine))
    fine_done = np.zeros((num_fine, num_fine), dtype=bool)  # the fine pixels that have their deflection angle

    # All the images grow together, so that every step is one call to the lens model for all of them
    todo = {i: mask for i, mask in todo.items() if mask.any()}

    while todo:
        needed = np.logical_or.reduce(list(todo.values())) & ~fine_done

        if needed.any():
            mx, my = np.nonzero(needed)
            fine_alpha_x[mx, my], fine_alpha_y[mx, my] = lens_model.alpha(fine_x[mx], fine_y[my], kwargs_lens)
            fine_done |= needed

        for i, mask in todo.items():
            px, py = np.nonzero(mask)
            x = fine_x[px] - ratios[i] * fine_alpha_x[px, py]
            y = fine_y[py] - ratios[i] * fine_alpha_y[px, py]
            is_inside = dist_hlr(sources[i], x, y) < max_radii[i]

            new = np.zeros((num_fine, num_fine), dtype=bool)
            new[px[is_inside], py[is_inside]] = True

            known[i] |= mask
            inside[i] |= new

            # Only the pixels that are inside make the image bigger, so only their neighbors are next
            todo[i] = binary_dilation(new) & ~known[i]

        todo = {i: mask for i, mask in todo.items() if mask.any()}

    # Outlines
    all_images = [[] for _ in sources]

    for i, mask in inside.items():
        if np.count_nonzero(mask) < 5:  # filter out very small images
            continue

        mask = binary_fill_holes(mask)

        if return_interior:
            labels, num_images = label(mask, structure=np.ones((3, 3)))

            for n in range(1, num_images + 1):
                xs, ys = np.nonzero(labels == n)
                all_images[i].append((fine_x[xs], fine_y[ys]))
        else:
            # padding makes sure the contours close at the edge
            for contour in find_contours(np.pad(mask.T, 1).astype(float), 0.5, fully_connected="high"):
                # undo the padding to get fine pixel coordinates, then convert to arcsecs
                all_images[i].append((fine_x[0] + (contour[:, 1] - 1) * fine_width, fine_y[0] + (contour[:, 0] - 1) * fine_width))

    return all_images

def _find_midline(arc, pix_width=0.1):
    """
    Arc: (X, Y). Must be interior, not boundary
    Returns (x, y, dist_edge) midline points arrays
    """

    (X, Y) = arc

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
    local_max = maximum_filter(
        distance,
        size=9
    )

    # Keep points close to the local maximum
    ridge = (
        (distance >= local_max - 2.0) & mask
    )

    # Lightly close small gaps
    if min(ridge.shape) >= 5:
        ridge = binary_closing(
            ridge,
            structure=np.ones((5, 5), dtype=bool),
            iterations=3
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

    :param arc: Tuple (X, Y) of X and Y coordinate arrays.
    :param pix_width: fine pixel width used in trace_finite_source_images

    :returns: (length, width, orientation)
    """

    (_, _, distedge) = _find_midline(arc, pix_width)
    W = 2 * np.mean(distedge)

    (X, Y) = arc

    cx, cy = np.mean(X), np.mean(Y)

    p1_idx = np.argmax(np.hypot(X - cx, Y - cy))
    p1x, p1y = X[p1_idx], Y[p1_idx]

    p2_idx = np.argmax(np.hypot(X - p1x, Y - p1y))
    p2x, p2y = X[p2_idx], Y[p2_idx]

    ang_vector = (p2x - p1x, p2y - p1y)
    L = np.hypot(*ang_vector)

    phi = np.abs(np.dot(ang_vector, (cx, cy)) / np.hypot(cx, cy) / np.hypot(*ang_vector))

    return L, W, phi