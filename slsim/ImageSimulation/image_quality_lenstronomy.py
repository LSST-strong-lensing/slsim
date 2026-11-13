"""Image-simulation helpers for observatory band configuration.

Band definitions and name resolution live in :mod:`slsim.Util.bandpass`.
The band helpers are re-exported here for backward compatibility.
"""

from slsim.Util.bandpass import (  # noqa: F401
    EUCLID_BAND_LIST,
    LSST_BAND_LIST,
    ROMAN_BAND_LIST,
    _ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON,
    _OBSERVATORY_REGISTRY,
    check_speclite_name,
    get_all_supported_bands,
    get_band_central_wavelength,
    get_band_effective_wavelength,
    get_band_log_wavelength_ratio,
    get_observatory,
    get_sncosmo_filtername,
    get_speclite_filtername,
    get_speclite_filternames,
    register_observatory,
)


def kwargs_single_band(band, observatory=None, **kwargs):
    """Return the lenstronomy single-band keyword dict for a given band.

    :param band: Imaging band name (e.g. ``'g'``, ``'F062'``, ``'VIS'``, etc.).
    :type band: str
    :param observatory: Observatory name.  When ``None`` the observatory registry is
        queried automatically based on *band*.
    :type observatory: str or None
    :param kwargs: Additional keyword arguments forwarded to the observatory
        class constructor (e.g. ``coadd_years``).
    :return: Configuration dict of imaging data for lenstronomy.
    :rtype: dict
    """
    if observatory is None:
        observatory = get_observatory(band)

    if observatory not in _OBSERVATORY_REGISTRY:
        raise ValueError(
            f"Observatory '{observatory}' is not registered. "
            f"Registered observatories: {list(_OBSERVATORY_REGISTRY.keys())}"
        )

    obs_class = _OBSERVATORY_REGISTRY[observatory]["class"]
    obs_instance = obs_class(band=band, **kwargs)
    return obs_instance.kwargs_single_band()
