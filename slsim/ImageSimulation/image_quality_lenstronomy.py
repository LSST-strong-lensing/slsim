from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid

_OBSERVATORY_REGISTRY: dict[str, dict] = {}


def register_observatory(name: str, observatory_class, bands: list, speclite_fmt=None):
    """Register a new observatory so all image_quality_lenstronomy functions
    recognise it automatically.

    :param name: Observatory name string (e.g. "MidEx"). Case-sensitive.
    :param observatory_class: Class whose constructor accepts ``band`` as its
        first keyword argument and exposes a ``kwargs_single_band()`` method.
    :param bands: List of band name strings owned by this observatory.
    :param speclite_fmt: Callable ``(band: str) -> str`` that returns the
        speclite filter name for a given band, or ``None`` if the observatory
        does not have speclite filters.
    """
    _OBSERVATORY_REGISTRY[name] = {
        "class": observatory_class,
        "bands": list(bands),
        "speclite_fmt": speclite_fmt,
    }


# ── Pre-registered observatories ────────────────────────────────────────────────────
register_observatory(
    name="LSST",
    observatory_class=LSST,
    bands=["u", "g", "r", "i", "z", "y"],
    speclite_fmt=lambda band: f"lsst2023-{band}",
)
register_observatory(
    name="Roman",
    observatory_class=Roman,
    bands=["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"],
    speclite_fmt=lambda band: f"Roman-{band}",
)
register_observatory(
    name="Euclid",
    observatory_class=Euclid,
    bands=["VIS"],
    speclite_fmt=lambda band: f"Euclid-{band}",
)


def _get_observatory_name_for_band(band: str) -> str:
    """Return the observatory name that owns *band*, searching the registry.

    :param band: Imaging band name.
    :raises ValueError: if no registered observatory claims the band.
    """
    for obs_name, info in _OBSERVATORY_REGISTRY.items():
        if band in info["bands"]:
            return obs_name
    raise ValueError(
        f"Band '{band}' is not recognised by any registered observatory. "
        f"Registered bands: { {o: i['bands'] for o, i in _OBSERVATORY_REGISTRY.items()} }"
    )


def kwargs_single_band(band, observatory=None, **kwargs):
    """Return the lenstronomy single-band keyword dict for a given band.

    :param band: Imaging band name (e.g. ``'g'``, ``'F062'``, ``'VIS'``, etc.).
    :type band: str
    :param observatory: Observatory name.  When ``None`` the registry is
        queried automatically based on *band*.
    :type observatory: str or None
    :param kwargs: Additional keyword arguments forwarded to the observatory
        class constructor (e.g. ``coadd_years``).
    :return: Configuration dict for lenstronomy.
    :rtype: dict
    """
    if observatory is None:
        observatory = _get_observatory_name_for_band(band)

    if observatory not in _OBSERVATORY_REGISTRY:
        raise ValueError(
            f"Observatory '{observatory}' is not registered. "
            f"Registered observatories: {list(_OBSERVATORY_REGISTRY.keys())}"
        )

    obs_class = _OBSERVATORY_REGISTRY[observatory]["class"]
    obs_instance = obs_class(band=band, **kwargs)
    return obs_instance.kwargs_single_band()


def get_observatory(band: str) -> str:
    """Return the observatory name for a given imaging band.

    Queries the registry; works for any registered observatory.

    :param band: Imaging band name.
    :raises ValueError: if the band does not belong to any observatory.
    """
    return _get_observatory_name_for_band(band)


def get_speclite_filtername(band: str) -> str:
    """Return the speclite filter name for a given band.

    :param band: Imaging band name.
    :raises ValueError: if the band is not registered or has no speclite
        filter.
    """
    obs_name = _get_observatory_name_for_band(band)
    fmt = _OBSERVATORY_REGISTRY[obs_name]["speclite_fmt"]
    if fmt is None:
        raise ValueError(
            f"Observatory '{obs_name}' (band '{band}') has no speclite filter registered."
        )
    return fmt(band)


def get_speclite_filternames(bands: list) -> list:
    """Return speclite filter names for a list of bands.

    :param bands: List of imaging band name strings.
    :return: List of speclite filter name strings in the same order.
    """
    return [get_speclite_filtername(band) for band in bands]


def get_all_supported_bands() -> list:
    """Return every band name currently registered across all observatories.

    :return: Flat list of band name strings.
    :rtype: list of str
    """
    all_bands = []
    for info in _OBSERVATORY_REGISTRY.values():
        all_bands.extend(info["bands"])
    return all_bands


# to make it work with the existing codebase
ALL_SUPPORTED_BANDS = get_all_supported_bands()
