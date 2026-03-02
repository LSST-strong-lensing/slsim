import numpy as np

# This file contains functions commonly used by the CosmosWebCatalog and HSTCosmosCatalog
# classes in Sources/SourceCatalogues/


def match_source(
    angular_size,
    physical_size,
    axis_ratio,
    n_sersic,
    processed_catalog,
    max_scale=1,
    match_n_sersic=False,
):
    """This function matches the parameters in source_dict to find a
    corresponding source in a source catalog. The parameters being
    matched are:

    1. physical size
    2. axis ratio
    3. n_sersic only if match_n_sersic is True

    Each parameter being matched is normalized so that the max is 1 and min is 0.
    Matching is then performed by selecting the nearest point in 2D space (3D if match_n_sersic is True).

    :param angular_size: desired angular size of the source [arcsec]
    :param physical_size: desired physical size of the source [kpc]
    :param axis_ratio: desired axis ratio
    :param sersic_angle: desired sersic angle
    :param processed_catalog: the returned object from calling process_catalog()
        See e.g. slsim/Sources/SourceCatalogues HSTCosmosCatalog or CosmosWebCatalog
    :param max_scale: The matched image will be scaled to have the desired angular size. Scaling up
        results in a more pixelated image. This input determines what the maximum up-scale factor is.
    :type max_scale: int or float
    :param match_n_sersic: determines whether to match based off of the sersic index as well.
        Since n_sersic is usually undefined and set to 1 in SLSim, this is set to False by default.
    :type match_n_sersic: bool
    :return: tuple(ndarray, float, float, int)
        This is the raw image matched from the catalog, the scale that the image needs to
        match angular size, the angle of rotation needed to match the desired e1 and e2, and the galaxy ID.
    """

    # Later, the matched image will be scaled to match angular size
    # We want to avoid upscaling to prevent pixelization of the image
    processed_catalog = processed_catalog[
        angular_size <= processed_catalog["angular_size"].data * max_scale
    ]
    if len(processed_catalog) == 0:
        return None

    physical_sizes = np.append(processed_catalog["physical_size"].data, physical_size)
    physical_sizes = normalize_features(physical_sizes, norm_type="minmax")

    axis_ratios = np.append(processed_catalog["axis_ratio"].data, axis_ratio)
    axis_ratios = normalize_features(axis_ratios, norm_type="minmax")

    distances = (physical_sizes[:-1] - physical_sizes[-1]) ** 2 + (
        axis_ratios[:-1] - axis_ratios[-1]
    ) ** 2

    if match_n_sersic:
        n_sersics = np.append(processed_catalog["sersic_index"].data, n_sersic)
        n_sersics = normalize_features(n_sersics, norm_type="minmax")
        distances += (n_sersics[:-1] - n_sersics[-1]) ** 2

    matched_source = processed_catalog[np.argmin(distances)]

    return matched_source


def normalize_features(data, norm_type="zscore", data_min=None, data_max=None):
    """Normalizes a 1D array of data.

    :param data: 1d array of data
    :param norm_type: string indicating the type of normalization to
        apply
    :param data_min: minimum value of the data (can be used to override
        the default scaling)
    :param data_max: maximum value of the data (can be used to override
        the default scaling)
    :return: normalized 1d array of data
    """

    data = np.array(data, dtype=float)

    if norm_type == "minmax":
        d_min = data_min if data_min is not None else np.nanmin(data)
        d_max = data_max if data_max is not None else np.nanmax(data)
        if d_max == d_min:  # Prevent division by zero
            return np.zeros_like(data)
        return (data - d_min) / (d_max - d_min)

    elif norm_type == "zscore":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std

    else:
        raise ValueError("Unsupported normalization type. Use 'minmax' or 'zscore'.")


def safe_value(val):
    """This function ensures that a value that we put into a pandas DataFrame
    is safe, i.e doesn't have mismatched datatypes.

    :param val: value to store in df
    :type val: string or float or list or array
    :return: safe value
    """
    if isinstance(val, np.ndarray):
        # Ensure native byte order
        if hasattr(val, "dtype") and val.dtype.byteorder not in ("=", "|"):
            val = val.astype(val.dtype.newbyteorder("="))
        # If array has one element, convert to float
        if val.size == 1:
            return float(val)
    elif isinstance(val, np.generic):
        return float(val)
    return val
