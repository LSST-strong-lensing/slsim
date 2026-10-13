"""Experimental constant sky subtraction for observed galaxy cutouts.

Finite valid residuals are floored at zero after subtraction. No tapering,
flux rescaling, or noise removal is performed. The
outer 20 percent of the cutout is assumed to contain usable sky. Inspect the
returned mask: extended galaxy wings may violate that assumption. Supply a
coverage mask for zero-filled missing data; zeros are not inherently invalid.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter, label, binary_dilation


def subtract_hst_catalog_background(image, matched_source):
    """Subtract the HST catalog sky level and floor valid residuals at zero.

    Use original catalog cutouts. NOISE_VARIANCE is not subtracted. The input
    array is not modified; nonfinite pixels are preserved.

    :param image: Two-dimensional original image in native pixel-value units.
    :type image: numpy.ndarray
    :param matched_source: Catalog row or mapping with a finite, unmasked
        scalar ``NOISE_MEAN`` in the same units as ``image``.
    :type matched_source: astropy.table.Row or dict[str, object]
    :return: ``(corrected, background, sky_mask, diagnostics)``. ``corrected``
        is a float array of the same shape, with finite values floored at zero;
        ``background`` is the catalog mean; ``sky_mask`` is None because no
        sky is estimated. Diagnostics contain ``method`` (str), ``background``
        (float), ``clipped_pixels`` (int), and ``valid_flux_before`` and
        ``valid_flux_after`` (float pixel sums, excluding nonfinite pixels).
    :rtype: tuple[numpy.ndarray, float, None, dict[str, object]]
    """
    try:
        value = matched_source["NOISE_MEAN"]
        if np.ma.is_masked(value):
            raise ValueError("masked NOISE_MEAN")
        background = float(value)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            "HST background subtraction requires valid NOISE_MEAN."
        ) from exc
    if not np.isfinite(background):
        raise ValueError("HST NOISE_MEAN must be finite.")
    data = np.array(image, dtype=float, copy=True)
    if data.ndim != 2:
        raise ValueError("Expected a 2D HST cutout.")
    valid = np.isfinite(data)
    corrected = data.copy()
    corrected[valid] = np.maximum(data[valid] - background, 0.0)
    diagnostics = {
        "method": "hst_catalog_noise_mean",
        "background": background,
        "clipped_pixels": int(np.count_nonzero(data[valid] - background < 0)),
        "valid_flux_before": float(data[valid].sum()),
        "valid_flux_after": float(corrected[valid].sum()),
    }
    return corrected, background, None, diagnostics


def subtract_cosmos_catalog_background(
    image,
    source_mask=None,
    coverage_mask=None,
    border_fraction=0.2,
    sigma=3.0,
    dilation=2,
):
    """Estimate COSMOS Web sky with sigma clipping, subtract it, and floor at zero.

    A central ellipse and dilated bright-source mask exclude galaxy light
    from the outer border. Two clipping passes use a median centre, MAD-based
    scatter, and at most ten iterations each. The final sky is the median
    of retained pixels. Extended wings can still bias this estimate.

    :param image: Original 2D cutout, at least 10 pixels per side. Not modified.
    :type image: numpy.ndarray
    :param source_mask: Same-shape mask; True excludes pixels from sky
        estimation only. Those pixels are still corrected. Default is None.
    :type source_mask: numpy.ndarray or None
    :param coverage_mask: Same-shape mask; True marks missing coverage.
        These pixels are excluded from estimation and preserved in the output.
        Nonfinite pixels are also preserved. Default is None.
    :type coverage_mask: numpy.ndarray or None
    :param border_fraction: Border width as a fraction of the shorter image
        dimension, strictly between 0 and 0.5. Default is 0.2; at least one pixel.
    :type border_fraction: float
    :param sigma: Positive finite clipping threshold and bright-source
        detection multiplier. Default is 3.0.
    :type sigma: float
    :param dilation: Nonnegative number of bright-mask dilation iterations.
        Default is 2; zero disables dilation.
    :type dilation: int
    :return: ``(corrected, background, sky_mask, diagnostics)``. The float
        corrected array has the input shape and nonnegative finite valid pixels.
        The boolean sky mask is True for pixels used to estimate the scalar
        background. Diagnostics contain background and sky RMS (float, native
        units), sky_pixels and clipped_pixels (int), sky_fraction (float),
        edge_median_before/after and edge_rms_about_zero_before/after (float),
        valid_flux_before/after (float pixel sums), and caution (str). After
        statistics describe the floored output; sky RMS describes scatter,
        not uncertainty in the background estimate.
    :rtype: tuple[numpy.ndarray, float, numpy.ndarray, dict[str, object]]
    """
    data = np.array(image, dtype=float, copy=True)
    if data.ndim != 2 or min(data.shape) < 10:
        raise ValueError("Expected a 2D cutout at least 10 pixels per side.")
    if not 0 < border_fraction < 0.5:
        raise ValueError("Require 0 < border_fraction < 0.5.")

    if (
        not np.isfinite(sigma)
        or sigma <= 0
        or not isinstance(dilation, int)
        or dilation < 0
    ):
        raise ValueError("Invalid sigma or dilation.")

    def checked_mask(mask):
        """Validate a mask against the enclosing image shape.

        :param mask: Exclusion mask, or None for an all-False mask.
        :type mask: numpy.ndarray or None
        :return: Boolean mask with the enclosing image shape.
        :rtype: numpy.ndarray
        """
        if mask is None:
            return np.zeros(data.shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError("Mask shape must match image shape.")
        return mask

    valid = np.isfinite(data) & ~checked_mask(coverage_mask)
    ny, nx = data.shape
    yy, xx = np.indices(data.shape)
    width = max(1, int(min(data.shape) * border_fraction))
    edge = (xx < width) | (xx >= nx - width) | (yy < width) | (yy >= ny - width)
    central = ((xx - (nx - 1) / 2) / (0.35 * nx)) ** 2 + (
        (yy - (ny - 1) / 2) / (0.35 * ny)
    ) ** 2 < 1
    candidates = valid & edge & ~central & ~checked_mask(source_mask)
    if candidates.sum() < 30:
        raise ValueError("Insufficient sky pixels; use a larger cutout.")
    preliminary = sigma_clip(
        data[candidates], sigma=sigma, maxiters=10, cenfunc="median", stdfunc="mad_std"
    )
    level = float(np.ma.median(preliminary))
    rms = float(np.ma.std(preliminary))
    bright = valid & (data > level + sigma * rms)
    if dilation:
        bright = binary_dilation(bright, iterations=dilation)
    candidates &= ~bright
    clipped = sigma_clip(
        data[candidates], sigma=sigma, maxiters=10, cenfunc="median", stdfunc="mad_std"
    )
    sky_mask = np.zeros(data.shape, dtype=bool)
    sky_mask[candidates] = ~np.ma.getmaskarray(clipped)
    if sky_mask.sum() < 30:
        raise ValueError("Insufficient unmasked sky; use a larger cutout.")
    background = float(np.median(data[sky_mask]))
    corrected = data.copy()
    corrected[valid] = np.maximum(data[valid] - background, 0.0)
    # Include all valid border pixels in the diagnostic, including contaminants.
    edge_values = data[valid & edge]
    diagnostics = {
        "background": background,
        "sky_rms": float(np.std(data[sky_mask])),
        "sky_pixels": int(sky_mask.sum()),
        "sky_fraction": float(sky_mask.sum() / valid.sum()),
        "edge_median_before": float(np.median(edge_values)),
        "edge_median_after": float(np.median(corrected[valid & edge])),
        "edge_rms_about_zero_before": float(np.sqrt(np.mean(edge_values**2))),
        "edge_rms_about_zero_after": float(
            np.sqrt(np.mean(corrected[valid & edge] ** 2))
        ),
        "clipped_pixels": int(np.count_nonzero(data[valid] - background < 0)),
        "valid_flux_before": float(data[valid].sum()),
        "valid_flux_after": float(corrected[valid].sum()),
        "caution": "Sky is estimated on this border; edge statistics are not independent validation. Inspect masks for galaxy wings. Noise remains.",
    }
    return corrected, background, sky_mask, diagnostics


def check_galaxy_edge(
    image, background, margin_fraction=0.1, peak_fraction=0.01, noise_threshold=3.0
):
    """Detect central source structure entering the cutout border.

    A 3x3 median-filtered, background-subtracted copy is used for detection
    only. Track the 8-connected component containing the brightest pixel in
    the central half of each image dimension. The threshold is the maximum
    of peak_fraction times that peak and noise_threshold times the border
    MAD scatter. No flooring is applied to this detection copy. Faint wings
    and disconnected neighbours may be missed; this is not a morphology guarantee.

    :param image: Original 2D cutout. The input is not modified.
    :type image: numpy.ndarray
    :param background: Finite constant sky level in the image pixel units.
    :type background: float
    :param margin_fraction: Width of the rejection border as a fraction of
        the shorter dimension, strictly between 0 and 0.5. Default is 0.1.
    :type margin_fraction: float
    :param peak_fraction: Fraction of central peak defining the relative
        threshold, strictly between 0 and 1. Default is 0.01.
    :type peak_fraction: float
    :param noise_threshold: Positive multiplier of border MAD scatter.
        Default is 3.0.
    :type noise_threshold: float
    :return: Dictionary with rejected (bool) and reason (str): accepted,
        edge_structure, undetected_source, or invalid_cutout. Valid cutouts
        also include threshold, peak, background (float), and margin_pixels
        (int). Non-2D, smaller-than-10-pixel, or nonfinite cutouts are rejected.
    :rtype: dict[str, object]
    """
    data = np.asarray(image, dtype=float)
    if (
        not 0 < margin_fraction < 0.5
        or not 0 < peak_fraction < 1
        or noise_threshold <= 0
    ):
        raise ValueError("Invalid edge detection thresholds.")
    if data.ndim != 2 or min(data.shape) < 10 or not np.isfinite(data).all():
        return {"rejected": True, "reason": "invalid_cutout"}
    smooth = median_filter(data - background, size=3)
    ny, nx = data.shape
    width = max(1, int(min(data.shape) * margin_fraction))
    edge = np.ones(data.shape, dtype=bool)
    edge[width:-width, width:-width] = False
    values = smooth[edge]
    noise = 1.4826 * np.median(np.abs(values - np.median(values)))
    center = smooth[ny // 4 : ny - ny // 4, nx // 4 : nx - nx // 4]
    iy, ix = np.unravel_index(np.argmax(center), center.shape)
    seed = (iy + ny // 4, ix + nx // 4)
    peak = float(smooth[seed])
    threshold = max(noise_threshold * noise, peak_fraction * peak)
    components, _ = label(smooth > threshold, structure=np.ones((3, 3)))
    component = components[seed]
    detected = component != 0
    touches = bool(np.any((components == component) & edge)) if detected else False
    return {
        "rejected": touches or not detected,
        "reason": (
            "edge_structure"
            if touches
            else ("accepted" if detected else "undetected_source")
        ),
        "threshold": float(threshold),
        "margin_pixels": width,
        "peak": peak,
        "background": float(background),
    }


def filter_edge_catalog(catalog, catalog_type, catalog_path, **edge_kwargs):
    """Filter all native templates before parameter matching.

    All four COSMOS Web bands must pass; HST uses its specified image HDU.
    The original table and FITS files are unchanged. This function does not
    cache results; CatalogSource caches them. Invalid images and COSMOS Web
    sky-estimation failures reject the template, while file and HST metadata
    errors propagate.

    :param catalog: Processed table. HST requires IDENT, GAL_FILENAME,
        GAL_HDU and NOISE_MEAN; COSMOS Web requires id. Other columns are retained.
    :type catalog: astropy.table.Table
    :param catalog_type: Either ``HST_COSMOS`` or ``COSMOS_WEB``.
    :type catalog_type: str
    :param catalog_path: Directory containing the catalog image FITS files.
    :type catalog_path: str or pathlib.Path
    :param edge_kwargs: Keyword overrides forwarded to check_galaxy_edge:
        margin_fraction, peak_fraction, and noise_threshold (float).
    :type edge_kwargs: dict[str, float]
    :return: ``(filtered_catalog, diagnostics)``. The new table contains only
        accepted rows, in original order, and may be empty. Diagnostics is a
        list in input-row order; each item has id (int), rejected (bool), and
        bands (list of detection dictionaries). Band order is the single HST
        HDU or F115W, F150W, F277W, F444W. Failed sky estimation has reason
        insufficient_sky; invalid arrays have reason invalid_cutout.
    :rtype: tuple[astropy.table.Table, list[dict[str, object]]]
    """
    if catalog_type not in ("HST_COSMOS", "COSMOS_WEB"):
        raise ValueError("Unsupported catalog type.")
    # Validate settings even for an empty catalog.
    check_galaxy_edge(np.zeros((10, 10)), 0, **edge_kwargs)
    accepted, diagnostics = [], []
    for row in catalog:
        hst = catalog_type == "HST_COSMOS"
        identifier = int(row["IDENT"] if hst else row["id"])
        name = (
            row["GAL_FILENAME"] if hst else f"COSMOSWeb_galaxy_{identifier}_image.fits"
        )
        bands = []
        with fits.open(Path(catalog_path) / name) as hdul:
            indices = [int(row["GAL_HDU"])] if hst else range(1, 5)
            for index in indices:
                img = hdul[index].data
                if (
                    img is None
                    or img.ndim != 2
                    or min(img.shape) < 10
                    or not np.isfinite(img).all()
                ):
                    bands.append({"rejected": True, "reason": "invalid_cutout"})
                    continue
                if hst:
                    background = subtract_hst_catalog_background(img, row)[1]
                else:
                    try:
                        background = subtract_cosmos_catalog_background(img)[1]
                    except ValueError:
                        bands.append({"rejected": True, "reason": "insufficient_sky"})
                        continue
                bands.append(check_galaxy_edge(img, background, **edge_kwargs))
        rejected = any(band["rejected"] for band in bands)
        accepted.append(not rejected)
        diagnostics.append({"id": identifier, "rejected": rejected, "bands": bands})
    return catalog[np.asarray(accepted, dtype=bool)], diagnostics
