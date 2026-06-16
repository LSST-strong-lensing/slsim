"""Utilities for displaying pre-simulated Euclid images.

The functions in this module are intended to be used after the physical image
simulation has already been performed with ``slsim.ImageSimulation.simulate_image``.
They convert Euclid VIS/Y/J(/H) image arrays into RGB products following the
image processing options illustrated in the Euclid Q1 Strong Lensing Discovery
Engine paper.
    See https://arxiv.org/pdf/2503.15324
    for specific details on the Euclid Q1 image processing steps.
    We offer a variety of options for procedures not mentioned in the paper,
    or for methods used in the paper that are not applicable to the simulated data.

The expected input order is ``[VIS, Y, J]`` or ``[VIS, Y, J, H]``. VIS is
treated as the high-resolution luminance channel, while the NISP bands provide
colour information and are resampled to the VIS image shape when necessary.
"""

import numpy as np
from scipy.ndimage import zoom
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid

EUCLID_Q1_ARCSINH_SCALE = {"VIS": 500.0, "Y": 1.0, "J": 0.5, "H": 0.25}


def euclid_rgb_from_image_list(
    image_list,
    colour="VIS_Y_J",
    stretch="mtf",
    black_percentile=1.0,
    white_percentile=99.85,
    arcsinh_scale=4.0,
    mtf_midtone=0.2,
    mtf_target_mean=0.2,
    mtf_region_size=100,
    use_luminance=True,
    luminance_method="mean",
    vis_red_fraction=0.35,
    vis_green_fraction=0.45,
    channel_gains=None,
    saturation=1.0,
):
    """Create a Euclid Q1-style RGB image from pre-simulated images.

    The colour options follow the Euclid Q1 display convention:

    - ``"VIS"``: grayscale VIS image.
    - ``"VIS_Y"``: red = Y, green = median(Y, VIS), blue = VIS.
    - ``"VIS_J"``: red = J, green = median(J, VIS), blue = VIS.
    - ``"VIS_Y_J"``: red = J, green = Y, blue = VIS.
    - ``"VIS_WEIGHTED_Y_J"``: red = weighted sum of J and VIS,
      green = weighted sum of Y and VIS, blue = VIS.
    - ``"VIS_WEIGHTED_Y_J_H"``: red = weighted sum of H and VIS,
      green = weighted sum of the Y/J mean and VIS, blue = VIS.

    When ``use_luminance`` is True, the final image luminance is set by the
    stretched VIS channel. This preserves the higher VIS spatial resolution while
    retaining colour information from the NISP bands.
        See https://arxiv.org/pdf/2503.15324
        for specific details on the Euclid Q1 image processing steps.

    :param image_list: images in order ``[VIS, Y, J]`` or ``[VIS, Y, J, H]``.
        NISP images may have lower native resolution than VIS and are resampled
        to the VIS shape when required.
    :type image_list: list[numpy.ndarray]
    :param colour: colour mapping to use. Supported values are ``"VIS"``,
        ``"VIS_Y"``, ``"VIS_J"``, ``"VIS_Y_J"``, and
        ``"VIS_WEIGHTED_Y_J"``, and ``"VIS_WEIGHTED_Y_J_H"``.
    :type colour: str
    :param stretch: display stretch to apply. Supported values are ``"linear"``,
        ``"arcsinh"``, and ``"mtf"``. ``"linear"`` applies only percentile
        normalisation without a nonlinear display transform.
    :type stretch: str
    :param black_percentile: percentile of each input image mapped to black before
        stretching.
    :type black_percentile: float
    :param white_percentile: percentile of each input image mapped to white before
        stretching.
    :type white_percentile: float
    :param arcsinh_scale: contrast parameter for the arcsinh stretch. A single
        float applies the same value to all bands. If None or ``"euclid_q1"``,
        use the Euclid Q1 values ``{"VIS": 500, "Y": 1, "J": 0.5}`` and the
        display fallback value ``{"H": 0.25}`` for the optional H band. A
        dictionary can override individual band values.
    :type arcsinh_scale: float or dict or str or None
    :param mtf_midtone: midtone transfer function parameter. Values below 0.5
        brighten faint structure; values above 0.5 darken it. If ``"auto"``,
        solve for the value that gives the central VIS region a mean of
        ``mtf_target_mean`` after MTF stretching.
    :type mtf_midtone: float or str
    :param mtf_target_mean: target mean of the central VIS region when
        ``mtf_midtone="auto"``.
    :type mtf_target_mean: float
    :param mtf_region_size: size in pixels of the central square region used for
        automatic MTF calibration. If the image is smaller, the full image is
        used.
    :type mtf_region_size: int
    :param use_luminance: if True, replace RGB luminance with the stretched VIS
        channel after colour construction.
    :type use_luminance: bool
    :param luminance_method: method used to estimate the current RGB luminance
        before replacing it with the VIS luminance. ``"mean"`` uses the simple
        channel average and gives a softer display. ``"rec709"`` uses standard
        RGB luminance weights.
    :type luminance_method: str
    :param vis_red_fraction: VIS fraction mixed into the red channel when using
        a ``"VIS_WEIGHTED_*"`` colour mode. For ``"VIS_WEIGHTED_Y_J"``, the red
        channel is ``(1 - vis_red_fraction) * J + vis_red_fraction * VIS``.
        For ``"VIS_WEIGHTED_Y_J_H"``, J is replaced by H.
    :type vis_red_fraction: float
    :param vis_green_fraction: VIS fraction mixed into the green channel when
        using a ``"VIS_WEIGHTED_*"`` colour mode. For ``"VIS_WEIGHTED_Y_J"``,
        the green channel is
        ``(1 - vis_green_fraction) * Y + vis_green_fraction * VIS``. For
        ``"VIS_WEIGHTED_Y_J_H"``, Y is replaced by the mean of Y and J.
    :type vis_green_fraction: float
    :param channel_gains: optional display-only multiplicative gains applied to
        the final ``(R, G, B)`` channels. ``None`` keeps the Euclid Q1 colour
        mapping unchanged. This can be useful for reducing VIS/blue dominance
        in purely simulated noisy images without changing the physical
        simulation.
    :type channel_gains: tuple[float, float, float] or None
    :param saturation: display-only saturation factor applied after
        ``channel_gains``. ``1`` leaves saturation unchanged, values below ``1``
        soften colour noise, and values above ``1`` increase colour contrast.
    :type saturation: float
    :return: RGB image with values clipped to ``[0, 1]`` and shape
        ``(ny, nx, 3)``.
    :rtype: numpy.ndarray
    """
    if len(image_list) < 1:
        raise ValueError("image_list must contain at least VIS image.")

    vis = np.asarray(image_list[0], dtype=float)
    y = np.asarray(image_list[1], dtype=float) if len(image_list) > 1 else None
    j = np.asarray(image_list[2], dtype=float) if len(image_list) > 2 else None
    h = np.asarray(image_list[3], dtype=float) if len(image_list) > 3 else None

    target_shape = vis.shape
    if y is not None and y.shape != target_shape:
        y = _resample_to_shape(y, target_shape)
    if j is not None and j.shape != target_shape:
        j = _resample_to_shape(j, target_shape)
    if h is not None and h.shape != target_shape:
        h = _resample_to_shape(h, target_shape)

    vis = _prepare_channel(vis, black_percentile, white_percentile)
    if y is not None:
        y = _prepare_channel(y, black_percentile, white_percentile)
    if j is not None:
        j = _prepare_channel(j, black_percentile, white_percentile)
    if h is not None:
        h = _prepare_channel(h, black_percentile, white_percentile)

    vis_stretched, y_stretched, j_stretched, h_stretched = _stretch_euclid_channels(
        vis=vis,
        y=y,
        j=j,
        h=h,
        stretch=stretch,
        arcsinh_scale=arcsinh_scale,
        mtf_midtone=mtf_midtone,
        mtf_target_mean=mtf_target_mean,
        mtf_region_size=mtf_region_size,
    )
    luminance = vis_stretched

    if colour == "VIS":
        rgb = np.dstack([luminance, luminance, luminance])

    elif colour == "VIS_Y":
        _require_channel(y, "Y", colour)
        green = _mixed_channel(
            vis,
            y,
            stretch=stretch,
            arcsinh_scale=arcsinh_scale,
            mtf_midtone=mtf_midtone,
            mtf_target_mean=mtf_target_mean,
            mtf_region_size=mtf_region_size,
            band="Y",
        )
        rgb = np.dstack(
            [
                y_stretched,
                green,
                vis_stretched,
            ]
        )

    elif colour == "VIS_J":
        _require_channel(j, "J", colour)
        green = _mixed_channel(
            vis,
            j,
            stretch=stretch,
            arcsinh_scale=arcsinh_scale,
            mtf_midtone=mtf_midtone,
            mtf_target_mean=mtf_target_mean,
            mtf_region_size=mtf_region_size,
            band="J",
        )
        rgb = np.dstack(
            [
                j_stretched,
                green,
                vis_stretched,
            ]
        )

    elif colour == "VIS_Y_J":
        _require_channel(y, "Y", colour)
        _require_channel(j, "J", colour)
        rgb = np.dstack(
            [
                j_stretched,
                y_stretched,
                vis_stretched,
            ]
        )

    elif colour == "VIS_WEIGHTED_Y_J":
        _require_channel(y, "Y", colour)
        _require_channel(j, "J", colour)
        red = _vis_weighted_channel(
            vis_channel=vis_stretched,
            colour_channel=j_stretched,
            vis_fraction=vis_red_fraction,
            channel_name="red",
        )
        green = _vis_weighted_channel(
            vis_channel=vis_stretched,
            colour_channel=y_stretched,
            vis_fraction=vis_green_fraction,
            channel_name="green",
        )
        rgb = np.dstack([red, green, vis_stretched])

    elif colour == "VIS_WEIGHTED_Y_J_H":
        _require_channel(y, "Y", colour)
        _require_channel(j, "J", colour)
        _require_channel(h, "H", colour)
        red = _vis_weighted_channel(
            vis_channel=vis_stretched,
            colour_channel=h_stretched,
            vis_fraction=vis_red_fraction,
            channel_name="red",
        )
        green = _vis_weighted_channel(
            vis_channel=vis_stretched,
            colour_channel=0.5 * (y_stretched + j_stretched),
            vis_fraction=vis_green_fraction,
            channel_name="green",
        )
        rgb = np.dstack([red, green, vis_stretched])

    else:
        raise ValueError(
            "colour must be 'VIS', 'VIS_Y', 'VIS_J', 'VIS_Y_J', "
            "'VIS_WEIGHTED_Y_J', or 'VIS_WEIGHTED_Y_J_H'."
        )

    if use_luminance and colour != "VIS":
        rgb = _apply_luminance(rgb, luminance, method=luminance_method)

    rgb = _apply_display_colour_balance(
        rgb,
        channel_gains=channel_gains,
        saturation=saturation,
    )

    return np.clip(rgb, 0, 1)


def euclid_nisp_num_pix_from_vis(num_pix_vis):
    """Calculate a NISP pixel count that covers the same field as a VIS image.

    :param num_pix_vis: number of pixels per axis in the VIS image.
    :type num_pix_vis: int
    :return: number of NISP pixels per axis needed to cover the VIS
        field of view.
    :rtype: int
    """
    vis_pixel_scale = Euclid(band="VIS").kwargs_single_band()["pixel_scale"]
    nisp_pixel_scale = Euclid(band="Y").kwargs_single_band()["pixel_scale"]

    fov = num_pix_vis * vis_pixel_scale
    return int(np.ceil(fov / nisp_pixel_scale))


def _resample_to_shape(image, target_shape):
    """Resample a 2D image to a target shape.

    :param image: input image.
    :type image: numpy.ndarray
    :param target_shape: target ``(ny, nx)`` shape.
    :type target_shape: tuple[int, int]
    :return: resampled image with shape ``target_shape``.
    :rtype: numpy.ndarray
    """
    scale_y = target_shape[0] / image.shape[0]
    scale_x = target_shape[1] / image.shape[1]
    resampled = zoom(image, (scale_y, scale_x), order=1)
    return _center_crop_or_pad(resampled, target_shape)


def _center_crop_or_pad(image, target_shape):
    """Centre-crop or zero-pad an image to a target shape.

    :param image: input image.
    :type image: numpy.ndarray
    :param target_shape: desired ``(ny, nx)`` shape.
    :type target_shape: tuple[int, int]
    :return: image centred in an array of shape ``target_shape``.
    :rtype: numpy.ndarray
    """
    out = np.zeros(target_shape, dtype=float)

    in_y, in_x = image.shape
    out_y, out_x = target_shape

    copy_y = min(in_y, out_y)
    copy_x = min(in_x, out_x)

    in_y0 = (in_y - copy_y) // 2
    in_x0 = (in_x - copy_x) // 2
    out_y0 = (out_y - copy_y) // 2
    out_x0 = (out_x - copy_x) // 2

    out[out_y0 : out_y0 + copy_y, out_x0 : out_x0 + copy_x] = image[
        in_y0 : in_y0 + copy_y,
        in_x0 : in_x0 + copy_x,
    ]
    return out


def _prepare_channel(image, black_percentile, white_percentile):
    """Normalise a single image channel to the interval ``[0, 1]``.

    :param image: input image channel.
    :type image: numpy.ndarray
    :param black_percentile: percentile mapped to 0.
    :type black_percentile: float
    :param white_percentile: percentile mapped to 1.
    :type white_percentile: float
    :return: normalised image channel.
    :rtype: numpy.ndarray
    """
    image = np.asarray(image, dtype=float)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    black = np.nanpercentile(image, black_percentile)
    white = np.nanpercentile(image, white_percentile)

    if not np.isfinite(white - black) or white <= black:
        return np.zeros_like(image)

    channel = (image - black) / (white - black)
    return np.clip(channel, 0, 1)


def _stretch_euclid_channels(
    vis,
    y,
    j,
    h,
    stretch,
    arcsinh_scale,
    mtf_midtone,
    mtf_target_mean,
    mtf_region_size,
):
    """Stretch VIS, Y, J, and optional H channels using display settings.

    The linear branch returns the percentile-normalised channels without an
    additional nonlinear transform. The arcsinh branch supports band-dependent Q
    values. By default these are the Euclid Q1 values: 500 for VIS, 1 for Y, and
    0.5 for J. The MTF branch supports the Euclid Q1 automatic midtone selection
    based on the central VIS image region.

    :param vis: normalised VIS channel.
    :type vis: numpy.ndarray
    :param y: normalised Y channel, or None if not provided.
    :type y: numpy.ndarray or None
    :param j: normalised J channel, or None if not provided.
    :type j: numpy.ndarray or None
    :param h: normalised H channel, or None if not provided.
    :type h: numpy.ndarray or None
    :param stretch: stretch name, either ``"linear"``, ``"arcsinh"``, or
        ``"mtf"``.
    :type stretch: str
    :param arcsinh_scale: arcsinh scale setting, passed to
        :func:`_arcsinh_scale_for_band`.
    :type arcsinh_scale: float or dict or None
    :param mtf_midtone: MTF midtone value, or ``"auto"``.
    :type mtf_midtone: float or str
    :param mtf_target_mean: target central-region mean for automatic MTF.
    :type mtf_target_mean: float
    :param mtf_region_size: central-region size in pixels for automatic MTF.
    :type mtf_region_size: int
    :return: stretched ``(VIS, Y, J, H)`` channels. Missing channels are returned
        as None.
    :rtype: tuple[numpy.ndarray, numpy.ndarray or None, numpy.ndarray or None,
        numpy.ndarray or None]
    """
    if stretch == "linear":
        return vis, y, j, h

    if stretch == "arcsinh":
        return (
            _arcsinh_stretch(vis, _arcsinh_scale_for_band(arcsinh_scale, "VIS")),
            (
                None
                if y is None
                else _arcsinh_stretch(y, _arcsinh_scale_for_band(arcsinh_scale, "Y"))
            ),
            (
                None
                if j is None
                else _arcsinh_stretch(j, _arcsinh_scale_for_band(arcsinh_scale, "J"))
            ),
            (
                None
                if h is None
                else _arcsinh_stretch(h, _arcsinh_scale_for_band(arcsinh_scale, "H"))
            ),
        )

    if stretch == "mtf":
        midtone = _resolve_mtf_midtone(
            mtf_midtone=mtf_midtone,
            reference_channel=vis,
            target_mean=mtf_target_mean,
            region_size=mtf_region_size,
        )
        return (
            _midtone_transfer_function(vis, midtone),
            None if y is None else _midtone_transfer_function(y, midtone),
            None if j is None else _midtone_transfer_function(j, midtone),
            None if h is None else _midtone_transfer_function(h, midtone),
        )

    raise ValueError("stretch must be 'linear', 'arcsinh', or 'mtf'.")


def _mixed_channel(
    vis,
    colour_channel,
    stretch,
    arcsinh_scale,
    mtf_midtone,
    mtf_target_mean,
    mtf_region_size,
    band,
):
    """Create the mixed green channel used by two-band colour modes.

    The Euclid Q1 paper specifies the two-band colour option as
    ``Y_E / median / I_E`` or ``J_E / median / I_E``. Here the median is
    computed from the normalised input channels before applying the nonlinear
    display stretch. This keeps channel mixing in the data layer and then applies
    the display transform once, which is the softer and more conventional choice
    for this RGB-display helper.

    :param vis: normalised VIS channel before stretching.
    :type vis: numpy.ndarray
    :param colour_channel: normalised Y or J channel before stretching.
    :type colour_channel: numpy.ndarray
    :param stretch: stretch name, either ``"arcsinh"`` or ``"mtf"``.
    :type stretch: str
    :param arcsinh_scale: arcsinh scale setting.
    :type arcsinh_scale: float or dict or str or None
    :param mtf_midtone: MTF midtone value, or ``"auto"``.
    :type mtf_midtone: float or str
    :param mtf_target_mean: target central-region mean for automatic MTF.
    :type mtf_target_mean: float
    :param mtf_region_size: central-region size for automatic MTF.
    :type mtf_region_size: int
    :param band: colour-channel band name, usually ``"Y"`` or ``"J"``.
    :type band: str
    :return: mixed green channel.
    :rtype: numpy.ndarray
    """
    mixed = np.median(np.dstack([colour_channel, vis]), axis=2)

    if stretch == "linear":
        return mixed

    if stretch == "arcsinh":
        mixed_scale = _arcsinh_scale_for_mixed_channel(arcsinh_scale, band)
        return _arcsinh_stretch(mixed, mixed_scale)

    if stretch == "mtf":
        midtone = _resolve_mtf_midtone(
            mtf_midtone=mtf_midtone,
            reference_channel=vis,
            target_mean=mtf_target_mean,
            region_size=mtf_region_size,
        )
        return _midtone_transfer_function(mixed, midtone)

    raise ValueError("stretch must be 'linear', 'arcsinh', or 'mtf'.")


def _vis_weighted_channel(
    vis_channel,
    colour_channel,
    vis_fraction,
    channel_name,
):
    """Mix VIS morphology into a NISP colour channel.

    This helper supports the simulation-friendly ``"VIS_WEIGHTED_Y_J"`` colour
    mode. It keeps the high-resolution VIS structure in the red and green
    channels while still retaining the NISP colour information.

    :param vis_channel: stretched or normalised VIS image.
    :type vis_channel: numpy.ndarray
    :param colour_channel: stretched or normalised Y/J image.
    :type colour_channel: numpy.ndarray
    :param vis_fraction: fraction of VIS to mix into the output channel.
    :type vis_fraction: float
    :param channel_name: display channel name used in error messages.
    :type channel_name: str
    :return: weighted channel
        ``(1 - vis_fraction) * colour_channel + vis_fraction * VIS``.
    :rtype: numpy.ndarray
    """
    if not 0 <= vis_fraction <= 1:
        raise ValueError(f"vis_{channel_name}_fraction must be between 0 and 1.")

    return (1 - vis_fraction) * colour_channel + vis_fraction * vis_channel


def _arcsinh_scale_for_band(arcsinh_scale, band):
    """Return the arcsinh Q value for a Euclid band.

    :param arcsinh_scale: if None, use Euclid Q1 defaults. If a number, use it
        for every band. If a dictionary, look up the requested band and fall back
        to the Euclid Q1 default when the band is absent.
    :type arcsinh_scale: float or dict or None
    :param band: Euclid band name, usually ``"VIS"``, ``"Y"``, or ``"J"``.
    :type band: str
    :return: arcsinh Q value for the band.
    :rtype: float
    """
    if arcsinh_scale is None or arcsinh_scale == "euclid_q1":
        return EUCLID_Q1_ARCSINH_SCALE[band]

    if isinstance(arcsinh_scale, dict):
        return arcsinh_scale.get(band, EUCLID_Q1_ARCSINH_SCALE[band])

    return float(arcsinh_scale)


def _arcsinh_scale_for_mixed_channel(arcsinh_scale, band):
    """Return the arcsinh Q value for a mixed VIS-plus-colour channel.

    :param arcsinh_scale: arcsinh scale setting.
    :type arcsinh_scale: float or dict or str or None
    :param band: colour-channel band name mixed with VIS.
    :type band: str
    :return: arcsinh Q value for the mixed channel.
    :rtype: float
    """
    if arcsinh_scale is None or arcsinh_scale == "euclid_q1":
        return np.median(
            [EUCLID_Q1_ARCSINH_SCALE["VIS"], EUCLID_Q1_ARCSINH_SCALE[band]]
        )

    if isinstance(arcsinh_scale, dict):
        if "median" in arcsinh_scale:
            return arcsinh_scale["median"]
        return np.median(
            [
                arcsinh_scale.get("VIS", EUCLID_Q1_ARCSINH_SCALE["VIS"]),
                arcsinh_scale.get(band, EUCLID_Q1_ARCSINH_SCALE[band]),
            ]
        )

    return float(arcsinh_scale)


def _resolve_mtf_midtone(
    mtf_midtone,
    reference_channel,
    target_mean,
    region_size,
):
    """Return the MTF midtone value, computing it automatically when requested.

    :param mtf_midtone: numeric midtone value or ``"auto"``.
    :type mtf_midtone: float or str
    :param reference_channel: normalised channel used for auto calibration.
    :type reference_channel: numpy.ndarray
    :param target_mean: target mean after applying MTF to the central region.
    :type target_mean: float
    :param region_size: central square size in pixels.
    :type region_size: int
    :return: MTF midtone parameter.
    :rtype: float
    """
    if mtf_midtone != "auto":
        return float(mtf_midtone)

    return _auto_mtf_midtone(
        reference_channel=reference_channel,
        target_mean=target_mean,
        region_size=region_size,
    )


def _auto_mtf_midtone(reference_channel, target_mean=0.2, region_size=100):
    """Find an MTF midtone value from a central-region target mean.

    The Euclid Q1 paper sets the MTF parameter automatically so that the
    central 100 x 100 pixels have a mean of 0.2 after stretching. This
    helper implements the same idea with a bisection search.

    :param reference_channel: normalised reference image, typically VIS.
    :type reference_channel: numpy.ndarray
    :param target_mean: desired central-region mean after MTF
        stretching.
    :type target_mean: float
    :param region_size: central square size in pixels. The full image is
        used when it is smaller than this region.
    :type region_size: int
    :return: MTF midtone parameter between 0 and 1.
    :rtype: float
    """
    if not 0 < target_mean < 1:
        raise ValueError("mtf_target_mean must be between 0 and 1.")

    region = _central_region(reference_channel, region_size)
    if not np.any(region > 0):
        return 0.5

    low = 1e-6
    high = 1 - 1e-6

    for _ in range(60):
        mid = 0.5 * (low + high)
        mean_mid = np.mean(_midtone_transfer_function(region, mid))

        if mean_mid > target_mean:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def _central_region(image, region_size):
    """Extract the central square region of an image.

    :param image: input image.
    :type image: numpy.ndarray
    :param region_size: desired central square size. If this exceeds an
        image dimension, that full dimension is used.
    :type region_size: int
    :return: central image region.
    :rtype: numpy.ndarray
    """
    if region_size <= 0:
        raise ValueError("mtf_region_size must be positive.")

    ny, nx = image.shape
    size_y = min(region_size, ny)
    size_x = min(region_size, nx)
    y0 = (ny - size_y) // 2
    x0 = (nx - size_x) // 2
    return image[y0 : y0 + size_y, x0 : x0 + size_x]


def _arcsinh_stretch(channel, scale):
    """Apply an arcsinh stretch to a normalised image channel.
        See https://iopscience.iop.org/article/10.1086/382245/pdf
        for specific details on the arcsinh stretch function.

    :param channel: normalised image channel.
    :type channel: numpy.ndarray
    :param scale: positive arcsinh contrast parameter.
    :type scale: float
    :return: stretched channel in approximately ``[0, 1]``.
    :rtype: numpy.ndarray
    """
    if scale <= 0:
        raise ValueError("arcsinh_scale must be positive.")
    channel = np.clip(channel, 0, None)
    return np.arcsinh(scale * channel) / np.arcsinh(scale)


def _midtone_transfer_function(channel, midtone):
    """Apply a midtone transfer function stretch.
        See https://arxiv.org/pdf/2503.15324
        for specific details on the midtone transfer function used in the Euclid Q1 figure.

    :param channel: normalised image channel.
    :type channel: numpy.ndarray
    :param midtone: midtone parameter between 0 and 1. A value of 0.5 is close to
        the identity mapping; lower values brighten the image.
    :type midtone: float
    :return: stretched channel.
    :rtype: numpy.ndarray
    """
    if not 0 < midtone < 1:
        raise ValueError("mtf_midtone must be between 0 and 1.")
    channel = np.clip(channel, 0, 1)
    denominator = ((2 * midtone - 1) * channel) - midtone
    return ((midtone - 1) * channel) / denominator


def _apply_luminance(rgb, luminance, method="mean", eps=1e-8):
    """Replace RGB luminance with a target luminance image.

    :param rgb: input RGB image with shape ``(ny, nx, 3)``.
    :type rgb: numpy.ndarray
    :param luminance: target luminance image with shape ``(ny, nx)``.
    :type luminance: numpy.ndarray
    :param method: method used to estimate the current RGB luminance.
        ``"mean"`` uses a simple channel average. ``"rec709"`` uses standard
        RGB luminance weights.
    :type method: str
    :param eps: small value used to avoid division by zero.
    :type eps: float
    :return: RGB image whose luminance follows ``luminance``.
    :rtype: numpy.ndarray
    """
    rgb = np.clip(rgb, 0, 1)
    luminance = np.clip(luminance, 0, 1)

    if method == "mean":
        current_luminance = np.mean(rgb, axis=-1)
    elif method == "rec709":
        current_luminance = (
            0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        )
    else:
        raise ValueError("luminance_method must be 'mean' or 'rec709'.")

    scale = luminance / (current_luminance + eps)
    return np.clip(rgb * scale[:, :, None], 0, 1)


def _apply_display_colour_balance(rgb, channel_gains=None, saturation=1.0):
    """Apply optional display-only RGB gains and saturation adjustment.

    :param rgb: input RGB image with shape ``(ny, nx, 3)``.
    :type rgb: numpy.ndarray
    :param channel_gains: optional multiplicative gains for ``(R, G, B)``.
    :type channel_gains: tuple[float, float, float] or None
    :param saturation: colour saturation multiplier.
    :type saturation: float
    :return: colour-balanced RGB image clipped to ``[0, 1]``.
    :rtype: numpy.ndarray
    """
    rgb = np.clip(rgb, 0, 1)

    if channel_gains is not None:
        gains = np.asarray(channel_gains, dtype=float)
        if gains.shape != (3,):
            raise ValueError("channel_gains must contain three values for R, G, B.")
        rgb = rgb * gains[None, None, :]

    if saturation < 0:
        raise ValueError("saturation must be non-negative.")
    if saturation != 1.0:
        grey = np.mean(rgb, axis=-1, keepdims=True)
        rgb = grey + saturation * (rgb - grey)

    return np.clip(rgb, 0, 1)


def _require_channel(channel, band, colour):
    """Validate that a colour mode has the required image channel.

    :param channel: channel array, or None when the image was not provided.
    :type channel: numpy.ndarray or None
    :param band: band name required by the colour mode.
    :type band: str
    :param colour: requested colour mode.
    :type colour: str
    :raises ValueError: if ``channel`` is None.
    """
    if channel is None:
        raise ValueError(f"colour='{colour}' requires {band} image in image_list.")
