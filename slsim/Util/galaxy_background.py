"""Experimental constant sky subtraction for observed galaxy cutouts.

No output-value clipping, tapering, flux rescaling, or noise removal is performed. The
outer 20 percent of the cutout is assumed to contain usable sky. Inspect the
returned mask: extended galaxy wings may violate that assumption. Supply a
coverage mask for zero-filled missing data; zeros are not inherently invalid.

Run this file with FITS_PATH --hdu 1 --output OUTPUT_DIRECTORY to write a
diagnostic PNG, JSON statistics, and a new FITS (never the input file).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter, label, binary_dilation


def subtract_hst_catalog_background(image, matched_source):
    """Subtract the HST catalog NOISE_MEAN without estimating galaxy wings.

    Returns corrected image, scalar level, None (no estimation mask),
    and diagnostics. Missing/invalid metadata raises rather than
    silently using the outskirts as sky. NOISE_VARIANCE is not
    subtracted. Input pixels, shape and negative residuals are preserved
    apart from the constant shift. Use on original catalog cutouts, not
    already corrected images.
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
    corrected[valid] -= background
    diagnostics = {
        "method": "hst_catalog_noise_mean",
        "background": background,
        "valid_flux_before": float(data[valid].sum()),
        "valid_flux_after": float(corrected[valid].sum()),
    }
    return corrected, background, None, diagnostics


def subtract_galaxy_background(
    image,
    source_mask=None,
    coverage_mask=None,
    border_fraction=0.2,
    sigma=3.0,
    dilation=2,
):
    """Return corrected image, scalar sky, usable-sky mask, and diagnostics.

    Masks are boolean arrays with True marking excluded pixels. Coverage
    pixels and nonfinite pixels are preserved, not shifted. Source masks
    exclude pixels from estimation only. A central ellipse and dilated
    bright sources are excluded before iterative sigma clipping.
    Background RMS is a descriptive scatter, not the uncertainty in the
    estimated sky.
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
    corrected[valid] -= background
    # Include all valid border pixels in the diagnostic, including contaminants.
    edge_values = data[valid & edge]
    diagnostics = {
        "background": background,
        "sky_rms": float(np.std(data[sky_mask])),
        "sky_pixels": int(sky_mask.sum()),
        "sky_fraction": float(sky_mask.sum() / valid.sum()),
        "edge_median_before": float(np.median(edge_values)),
        "edge_median_after": float(np.median(edge_values - background)),
        "edge_rms_about_zero_before": float(np.sqrt(np.mean(edge_values**2))),
        "edge_rms_about_zero_after": float(
            np.sqrt(np.mean((edge_values - background) ** 2))
        ),
        "valid_flux_before": float(data[valid].sum()),
        "valid_flux_after": float(corrected[valid].sum()),
        "caution": "Sky is estimated on this border; edge statistics are not independent validation. Inspect masks for galaxy wings. Noise remains.",
    }
    return corrected, background, sky_mask, diagnostics


def check_galaxy_edge(
    image, background, margin_fraction=0.1, peak_fraction=0.01, noise_threshold=3.0
):
    """Flag detected central structure entering a border margin.

    Experimental detection on a 3x3 median-filtered copy only. Track the
    connected component containing the brightest pixel in the central
    half of the stamp. Faint structure below the threshold can be
    missed; this is not a guarantee of complete morphology. Invalid
    stamps are rejected. The original image is never masked, smoothed,
    or cropped.
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
    """Read and screen native templates before any parameter matching.

    Returns a new table and per-template diagnostics. Missing files or
    HST metadata errors propagate; unusable images/sky are rejected. No
    FITS is modified. All COSMOS Web bands must pass. Callers may cache
    this result.
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
                        background = subtract_galaxy_background(img)[1]
                    except ValueError:
                        bands.append({"rejected": True, "reason": "insufficient_sky"})
                        continue
                bands.append(check_galaxy_edge(img, background, **edge_kwargs))
        rejected = any(band["rejected"] for band in bands)
        accepted.append(not rejected)
        diagnostics.append({"id": identifier, "rejected": rejected, "bands": bands})
    return catalog[np.asarray(accepted, dtype=bool)], diagnostics


def compare_fits(path, hdu, output):
    """Save a reproducible diagnostic for one FITS image extension."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, header = fits.getdata(path, hdu, header=True)
    corrected, background, mask, stats = subtract_galaxy_background(data)
    stem = f"{path.stem}_hdu{hdu}"
    stats.update(
        input_file=str(path.resolve()), hdu=hdu, band=header.get("EXTNAME", "unknown")
    )
    finite = np.isfinite(data)
    scale = max(stats["sky_rms"], np.finfo(float).eps)
    vmin, vmax = -2 * scale, max(5 * scale, float(np.percentile(data[finite], 95)))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    pad = max(3, int(min(data.shape) * 0.08))
    for ax, values, title in zip(
        axes[:2], [data, corrected], ["Original", "Background subtracted"]
    ):
        shown = ax.imshow(
            np.pad(values, pad), origin="lower", cmap="magma", vmin=vmin, vmax=vmax
        )
        ax.set_title(title + " (zero padded)")
        ax.set_axis_off()
    fig.colorbar(shown, ax=list(axes[:2]), shrink=0.7, label="Native pixel units")
    axes[2].imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("White: pixels used for sky")
    axes[2].set_axis_off()
    axes[3].plot(np.nanmedian(data, axis=0), label="Before")
    axes[3].plot(np.nanmedian(corrected, axis=0), label="After")
    axes[3].axhline(0, color="black", lw=0.7)
    axes[3].set(xlabel="Column", ylabel="Median pixel value")
    axes[3].legend()
    fig.suptitle(
        f"{path.name} | HDU {hdu} | sky={background:.4g}, RMS={scale:.4g}", fontsize=10
    )
    fig.savefig(output / f"{stem}.png", dpi=150)
    plt.close(fig)
    header["BGSUB"] = (background, "Experimental constant background removed")
    # Deliberately refuse to overwrite any previous FITS result.
    fits.HDUList(
        [
            fits.PrimaryHDU(corrected, header=header),
            fits.ImageHDU(mask.astype(np.uint8), name="SKYMASK"),
        ]
    ).writeto(output / f"{stem}_bgsub.fits", overwrite=False)
    (output / f"{stem}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_path")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_fits(args.fits_path, args.hdu, args.output), indent=2))
