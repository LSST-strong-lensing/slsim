import numpy as np


def default_reference_band(source_dict, default="i"):
    """Select the available magnitude band nearest a preferred reference band.

    Bands are inferred from ``mag_<band>`` keys in ``source_dict``. The band
    closest to ``default`` in logarithmic effective-wavelength space is
    returned, matching the model's power-law wavelength dependence. If no
    magnitude band is available, ``default`` is returned unchanged.

    :param source_dict: Source parameters that may contain magnitude entries
        such as ``mag_g``, ``mag_i``, or ``mag_F158``.
    :type source_dict: dict or mapping
    :param default: Preferred reference band and fallback when no magnitude
        bands are available.
    :type default: str
    :return: Available band with the smallest absolute
        ``log(lambda_band / lambda_default)``, or ``default`` when none exists.
    :rtype: str
    """
    from slsim.ImageSimulation.image_quality_lenstronomy import (
        get_band_log_wavelength_ratio,
    )

    available_bands = [
        key.replace("mag_", "", 1)
        for key in source_dict
        if isinstance(key, str) and key.startswith("mag_")
    ]
    if not available_bands:
        return default

    positions = [
        abs(get_band_log_wavelength_ratio(band=band, reference_band=default))
        for band in available_bands
    ]
    return available_bands[int(np.argmin(positions))]


def component_weights_for_band(
    base_weights,
    band,
    color_gradient=None,
    source_dict=None,
    default_reference="i",
):
    """Return band-dependent component weights from local SED slopes.

    This is a lightweight, effective-wavelength approximation to chromatic
    light components. Instead of integrating a full stellar-population SED
    through each bandpass, each component is assigned a local power-law SED,
    ``S_k(lambda) proportional lambda**alpha_k``, evaluated at the central
    wavelength of the requested band. The reference-band component weights are
    then reweighted as

    ``w_k(b) = w_k(ref) * (lambda_b/lambda_ref)**alpha_k / normalization``.

    The approximation is intended to introduce controlled colour gradients in
    analytic multi-component light profiles; it is not a replacement for a
    stellar population synthesis model.

    Ref:
        Hogg et al. 2002, "The K correction", astro-ph/0210394:
            broadband fluxes are formally filter-response weighted SED integrals;
            this function uses the corresponding effective-wavelength limit.
        Conroy 2013, ARA&A, 51, 393:
            review of full stellar-population SED modelling, useful context for
            what is intentionally omitted by this lightweight approximation.
        La Barbera et al. 2005, MNRAS, 358, 1116; La Barbera & de Carvalho 2009,
        ApJ, 699, L76:
            observational motivation for radial colour gradients in galaxies.
    """
    weights = np.asarray(base_weights, dtype=float)
    weights = weights / np.sum(weights)

    if band is None or color_gradient is None:
        return tuple(float(weight) for weight in weights)
    if not isinstance(color_gradient, dict):
        raise ValueError("color_gradient must be a dictionary or None.")

    slopes = color_gradient.get(
        "component_spectral_slopes", color_gradient.get("sed_slopes")
    )
    if slopes is None:
        return tuple(float(weight) for weight in weights)

    slopes = np.asarray(slopes, dtype=float)
    if slopes.shape != weights.shape:
        raise ValueError(
            "color_gradient['component_spectral_slopes'] must match the "
            "number of components."
        )

    reference_band = color_gradient.get("reference_band")
    if reference_band is None:
        reference_band = default_reference_band(
            source_dict or {}, default=default_reference
        )

    min_weight = float(color_gradient.get("min_weight", 1e-4))
    if not 0 <= min_weight < 1 / len(weights):
        raise ValueError(
            "color_gradient['min_weight'] must be in [0, 1 / n_components)."
        )

    from slsim.ImageSimulation.image_quality_lenstronomy import (
        get_band_effective_wavelength,
    )

    wavelength = get_band_effective_wavelength(band)
    reference_wavelength = get_band_effective_wavelength(reference_band)
    if reference_wavelength <= 0:
        raise ValueError("The reference band wavelength must be positive.")

    sed_factors = (wavelength / reference_wavelength) ** slopes
    sed_weights = weights * sed_factors
    sed_weights = sed_weights / np.sum(sed_weights)
    sed_weights = np.clip(sed_weights, min_weight, 1 - min_weight)
    sed_weights = sed_weights / np.sum(sed_weights)
    return tuple(float(weight) for weight in sed_weights)


def attach_foreground_deflector_color_gradient(
    galaxy_table,
    color_gradient,
    component_weights=(0.4, 0.6),
):
    """Attach opt-in foreground colour-gradient columns to a deflector table.

    The resulting columns are consumed by the standard
    ``Source``/``DoubleSersic`` light-model path to split foreground light into
    two chromatic Sersic components.
    The operation is in-place and returns ``galaxy_table`` for convenience.

    :param galaxy_table: galaxy/deflector table to annotate
    :param color_gradient: dictionary with ``component_spectral_slopes`` and
     optional foreground component settings
    :param component_weights: two reference-band flux weights for the Sersic
     components
    :return: annotated galaxy table
    """
    if color_gradient is None:
        return galaxy_table
    if not isinstance(color_gradient, dict):
        raise ValueError("color_gradient must be a dictionary or None.")

    weights = np.asarray(component_weights, dtype=float)
    if weights.shape != (2,):
        raise ValueError("component_weights must contain two values.")
    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("component_weights must be non-negative with positive sum.")
    weights = weights / np.sum(weights)

    galaxy_table["color_gradient"] = [
        dict(color_gradient) for _ in range(len(galaxy_table))
    ]
    galaxy_table["w0"] = np.full(len(galaxy_table), weights[0])
    galaxy_table["w1"] = np.full(len(galaxy_table), weights[1])
    return galaxy_table


def radial_color_gradient_image(
    image,
    band,
    color_gradient,
    angular_size,
    pixel_scale,
    default_reference="F814W",
):
    """Apply a d(color)/dlog10(r) gradient to an image and preserve flux.

    See https://arxiv.org/pdf/1006.4056 for details.
    """
    if band is None or color_gradient is None:
        return image
    if not isinstance(color_gradient, dict):
        raise ValueError("color_gradient must be a dictionary or None.")

    grad_color = float(
        color_gradient.get("grad_color", color_gradient.get("gradient", 0.0))
    )
    if grad_color == 0:
        return image

    reference_band = color_gradient.get("reference_band") or default_reference
    from slsim.ImageSimulation.image_quality_lenstronomy import (
        get_band_log_wavelength_ratio,
    )

    band_offset = get_band_log_wavelength_ratio(
        band=band, reference_band=reference_band
    )
    if band_offset == 0:
        return image

    image = np.asarray(image, dtype=float)
    y_grid, x_grid = np.indices(image.shape, dtype=float)
    center_x = (image.shape[1] - 1) / 2
    center_y = (image.shape[0] - 1) / 2
    radius = np.hypot(x_grid - center_x, y_grid - center_y)
    half_light_radius_pixels = angular_size / pixel_scale
    radius = np.maximum(radius, 0.5)
    radius_ratio = radius / max(half_light_radius_pixels, 0.5)

    delta_mag = band_offset * grad_color * np.log10(radius_ratio)
    chromatic_image = image * 10 ** (-0.4 * delta_mag)
    original_flux = np.sum(image)
    chromatic_flux = np.sum(chromatic_image)
    if chromatic_flux != 0:
        chromatic_image *= original_flux / chromatic_flux
    return chromatic_image
