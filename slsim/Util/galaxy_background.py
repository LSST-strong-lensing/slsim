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


def subtract_catalog_background(image, noise_data, mean_key="NOISE_MEAN"):
    """Subtract a supplied background mean.

    When the catalog metadata provides background noise data, this function 
    reads the corresponding parameters and subtracts the background noise 
    from the image.

    :param image: Two-dimensional original image in native pixel-value units.
    :type image: numpy.ndarray
    :param noise_data: Catalog row or dictionary containing the background mean.
    :type noise_data: astropy.table.Row or dict[str, object]
    :param mean_key: Field containing a finite scalar mean in image units.
    :type mean_key: str
    :return: Corrected float image, background mean, and None (no sky mask).
        Finite output pixels are nonnegative. The input is unchanged.
    :rtype: tuple[numpy.ndarray, float, None]
    """
    try:
        value = noise_data[mean_key]
        if np.ma.is_masked(value):
            raise ValueError("masked NOISE_MEAN")
        if np.ndim(value) != 0:
            raise ValueError("Noise mean must be scalar.")
        background = float(value)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            "Background subtraction requires a valid scalar noise mean."
        ) from exc
    if not np.isfinite(background):
        raise ValueError("Noise mean must be finite.")
    data = np.array(image, dtype=float, copy=True)
    if data.ndim != 2:
        raise ValueError("Expected a 2D cutout.")
    valid = np.isfinite(data)
    np.subtract(data, background, out=data, where=valid)
    np.maximum(data, 0.0, out=data, where=valid)
    return data, background, None


def subtract_estimated_background(
    image,
    source_mask=None,
    coverage_mask=None,
    border_fraction=0.2,
    sigma=3.0,
    dilation=2,
):
    """Estimate background noise from the image, subtract it.

    When the catalog metadata does not provide background noise data, 
    this function will estimate the background noise from pixels near 
    the image edges.
    Exclude the central region and bright sources. Use 3-sigma clipping
    to remove pixels whose values differ from the median by more than
    three times the estimated scatter. Repeat until no more pixels are
    removed, up to ten iterations per pass. Use the median of the
    remaining pixels as the background noise.

    :param image: Original 2D cutout, at least 10 pixels per side. Not modified.
    :type image: numpy.ndarray
    :param source_mask: Same-shape mask; True excludes pixels from background
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
    :return: Corrected float image, background mean, and boolean sky mask.
        True mask pixels were used to estimate the background.
    :rtype: tuple[numpy.ndarray, float, numpy.ndarray]
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
        """Check that the mask has the same shape as the image.

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
    np.subtract(data, background, out=data, where=valid)
    np.maximum(data, 0.0, out=data, where=valid)
    return data, background, sky_mask


def catalog_noise_data(matched_source, noise_kwargs=None, band_index=0, band_count=1):
    """Read the background mean for the current band from the catalog or user settings.

    :param matched_source: Catalog row, dictionary, or None.
    :type matched_source: astropy.table.Row or dict or None
    :param noise_kwargs: Optional ``noise_mean`` value or list in native-band
        order, or ``mean_key`` naming a catalog column. A scalar applies to
        every band. Explicit values take priority. Without options, look for
        ``NOISE_MEAN`` then ``noise_mean``.
    :type noise_kwargs: dict or None
    :param band_index: Zero-based native-band index.
    :type band_index: int
    :param band_count: Number of native image bands.
    :type band_count: int
    :return: Dictionary with ``noise_mean``, or None if no mean is available.
        Missing explicitly requested columns and invalid options raise ValueError.
    :rtype: dict or None
    """
    options = {} if noise_kwargs is None else noise_kwargs
    if not isinstance(options, dict) or set(options) - {"noise_mean", "mean_key"}:
        raise ValueError("noise_kwargs accepts noise_mean and mean_key only.")
    if "noise_mean" in options:
        value = options["noise_mean"]
    else:
        keys = (
            [options["mean_key"]]
            if "mean_key" in options
            else ["NOISE_MEAN", "noise_mean"]
        )
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("mean_key must be a column name.")
        for key in keys:
            try:
                value = matched_source[key]
                break
            except (KeyError, TypeError, IndexError):
                continue
        else:
            if "mean_key" in options:
                raise ValueError("Requested noise mean column is missing.")
            return None
    if np.ndim(value) != 0:
        if np.ndim(value) != 1 or len(value) != band_count:
            raise ValueError(
                "noise_mean must be scalar or have one value per native band."
            )
        value = value[band_index]
    return {"noise_mean": value}


def subtract_source_background(
    image, matched_source=None, noise_kwargs=None, band_index=0, band_count=1
):
    """Use a supplied background noise when available; 
    otherwise estimate it from the image.

    :param image: Original two-dimensional image.
    :type image: numpy.ndarray
    :param matched_source: Catalog metadata.
    :type matched_source: astropy.table.Row or dict or None
    :param noise_kwargs: Mean values or column name; see catalog_noise_data.
    :type noise_kwargs: dict or None
    :param band_index: Zero-based native-band index.
    :type band_index: int
    :param band_count: Number of native image bands.
    :type band_count: int
    :return: Corrected image, background mean, and sky mask (None for a supplied mean).
    :rtype: tuple[numpy.ndarray, float, numpy.ndarray or None]
    """
    noise = catalog_noise_data(matched_source, noise_kwargs, band_index, band_count)
    if noise is not None:
        return subtract_catalog_background(image, noise, mean_key="noise_mean")
    return subtract_estimated_background(image)


def check_galaxy_edge(
    image, background, margin_fraction=0.1, peak_fraction=0.01, noise_threshold=3.0
):
    """Check whether the central source extends into the image border.

    Subtract the background and smooth a copy of the image.
    Find the brightest pixel in the central region, then identify
    the connected pixels above the detection threshold around it.
    Reject the image if this structure reaches the border or no
    central source is detected.

    :param image: Original 2D cutout. The input is not modified.
    :type image: numpy.ndarray
    :param background: Finite constant sky level in the image pixel
        units.
    :type background: float
    :param margin_fraction: Width of the rejection border as a fraction
        of the shorter dimension, strictly between 0 and 0.5. Default is
        0.1.
    :type margin_fraction: float
    :param peak_fraction: Fraction of central peak defining the relative
        threshold, strictly between 0 and 1. Default is 0.01.
    :type peak_fraction: float
    :param noise_threshold: Positive multiplier of border MAD scatter.
        Default is 3.0.
    :type noise_threshold: float
    :return: Dictionary with rejected (bool) and reason (str): accepted,
        edge_structure, undetected_source, or invalid_cutout. Valid
        cutouts also include threshold, peak, background (float), and
        margin_pixels (int). Non-2D, smaller-than-10-pixel, or nonfinite
        cutouts are rejected.
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


def filter_edge_catalog(
    catalog, catalog_type, catalog_path, noise_kwargs=None, **edge_kwargs
):
    """Remove sources whose central structure reaches the image border
    before matching.

    :param catalog: Processed table. HST requires IDENT, GAL_FILENAME,
        GAL_HDU; COSMOS Web requires id. Other columns are retained.
    :type catalog: astropy.table.Table
    :param catalog_type: Either ``HST_COSMOS`` or ``COSMOS_WEB``.
    :type catalog_type: str
    :param catalog_path: Directory containing the catalog image FITS files.
    :type catalog_path: str or pathlib.Path
    :param noise_kwargs: Mean values or column name; see catalog_noise_data.
    :type noise_kwargs: dict or None
    :param edge_kwargs: Keyword overrides forwarded to check_galaxy_edge:
        margin_fraction, peak_fraction, and noise_threshold (float).
    :type edge_kwargs: dict[str, float]
    :return: Catalog containing only sources that pass all native bands,
        in the original order. The result may be empty.
    :rtype: astropy.table.Table
    """
    if catalog_type not in ("HST_COSMOS", "COSMOS_WEB"):
        raise ValueError("Unsupported catalog type.")
    # Validate settings even for an empty catalog.
    check_galaxy_edge(np.zeros((10, 10)), 0, **edge_kwargs)
    accepted = []
    for row in catalog:
        hst = catalog_type == "HST_COSMOS"
        identifier = int(row["IDENT"] if hst else row["id"])
        name = (
            row["GAL_FILENAME"] if hst else f"COSMOSWeb_galaxy_{identifier}_image.fits"
        )
        bands = []
        with fits.open(Path(catalog_path) / name, memmap=False) as hdul:
            indices = [int(row["GAL_HDU"])] if hst else range(1, 5)
            for band_index, index in enumerate(indices):
                img = hdul[index].data
                if (
                    img is None
                    or img.ndim != 2
                    or min(img.shape) < 10
                    or not np.isfinite(img).all()
                ):
                    bands.append(True)
                    continue
                noise = catalog_noise_data(row, noise_kwargs, band_index, len(indices))
                if noise is not None:
                    background = subtract_catalog_background(
                        img, noise, mean_key="noise_mean"
                    )[1]
                else:
                    try:
                        background = subtract_estimated_background(img)[1]
                    except ValueError:
                        bands.append(True)
                        continue
                bands.append(check_galaxy_edge(img, background, **edge_kwargs)["rejected"])
        rejected = any(bands)
        accepted.append(not rejected)
    return catalog[np.asarray(accepted, dtype=bool)]
