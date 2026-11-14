"""Bandpass conventions shared across SLSim.

Short SLSim band names are the public API used by source magnitude columns
such as ``mag_<band>`` and ``ps_mag_<band>``. The same name is resolved to a
speclite filter or sncosmo bandpass in image simulation, supernova
magnitudes, accretion-disk reprocessing and AGN/SN morphology.

Resolution order:

1. A band registered by an observatory is mapped through that observatory's
   ``speclite_fmt`` and ``sncosmo_fmt`` callables.
2. A raw speclite or sncosmo filter name is passed through unchanged.
3. Roman speclite filters are configured lazily on first use.

Supported default bands and the curves they use:

- LSST: ``u``, ``g``, ``r``, ``i``, ``z``, ``y``;
  speclite ``lsst2023-<band>``, sncosmo ``lsst<band>``.
- Roman: ``F062``, ``F087``, ``F106``, ``F129``, ``F158``, ``F184``,
  ``F146``, ``F213``; speclite ``Roman-<band>``, sncosmo ``<band>``.
- Euclid: ``VIS``, ``Y``, ``J``, ``H``; speclite ``Euclid-<band>``,
  sncosmo ``euclid<band>``.

Additional observatories can be registered with :func:`register_observatory`.
Note that the speclite and sncosmo curves for the same survey band are not
guaranteed to be identical; this module follows the curves already used by
each downstream module.
"""

import numpy as np
import sncosmo
import speclite.filters

from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid

_OBSERVATORY_REGISTRY = {}


# Default options in SLSim
ROMAN_BAND_LIST = ["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"]
LSST_BAND_LIST = ["u", "g", "r", "i", "z", "y"]
EUCLID_BAND_LIST = ["VIS", "Y", "J", "H"]

# Ancillary bandpasses that are used by catalog sources but are not imaging
# observatories registered below.
_ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON = {"F814W": 0.805}


def check_speclite_name(band):
    """Check whether the raw band name is a valid speclite filter.

    Returns the band name if valid, otherwise returns None. This is the
    default ``speclite_fmt`` for observatories whose band names match their
    speclite filter names.

    :param band: band name to check
    :type band: str
    :return: band name if it is a valid speclite filter, otherwise None
    :rtype: str or None
    """
    try:
        speclite.filters.load_filter(band)
        return band
    except ValueError:
        return None


def register_observatory(
    name: str,
    observatory_class,
    bands: list,
    speclite_fmt=check_speclite_name,
    sncosmo_fmt=None,
    effective_wavelengths=None,
):
    """Register a new observatory and its band conventions.

    This allows external or user-defined observatories (e.g., ``"MidEx"``)
    to be recognized by :func:`get_observatory`, :func:`get_speclite_filtername`
    and :func:`kwargs_single_band`.

    :param name: identifier for the observatory (e.g., ``"MidEx"``)
    :type name: str
    :param observatory_class: class defining the observatory configuration.
        Similar to the ``lenstronomy.SimulationAPI.ObservationConfig``
        classes, it must accept ``band`` as a keyword argument and expose a
        ``kwargs_single_band()`` method.
    :type observatory_class: type
    :param bands: band names owned by this observatory
    :type bands: list[str]
    :param speclite_fmt: callable mapping a band name to its speclite
        filter name (e.g., ``lambda b: f"MidEx-{b}"``). Set to ``None`` when
        the observatory has no speclite filters.
    :type speclite_fmt: callable or None
    :param sncosmo_fmt: callable mapping a band name to its sncosmo name.
        Set to ``None`` to use the raw band name.
    :type sncosmo_fmt: callable or None
    :param effective_wavelengths: optional mapping from band names to
        throughput-weighted effective wavelengths in microns. Use this when
        the responses are not available through ``speclite_fmt``.
    :type effective_wavelengths: dict or None

    Example:
    --------
    .. code-block:: python

        import copy
        import lenstronomy.Util.util as util
        import slsim.Util.bandpass as bandpass

        custom_band_obs = {
            "A": {"exposure_time": 90.0, "sky_brightness": 22.0},
            "B": {"exposure_time": 90.0, "sky_brightness": 21.5},
        }

        class CustomObs:
            def __init__(self, band="A", **kwargs):
                if band not in custom_band_obs:
                    raise ValueError(f"Band '{band}' not supported.")
                self.obs = copy.deepcopy(custom_band_obs[band])
                self.camera = {
                    "read_noise": 5.0,
                    "pixel_scale": 0.15,
                    "ccd_gain": 2.0,
                }

            def kwargs_single_band(self):
                return util.merge_dicts(self.camera, self.obs)

        bandpass.register_observatory(
            name="CustomObs",
            observatory_class=CustomObs,
            bands=list(custom_band_obs.keys()),
            speclite_fmt=lambda b: f"CustomObs-{b}",
            sncosmo_fmt=lambda b: f"customobs{b}",
        )
    """
    _OBSERVATORY_REGISTRY[name] = {
        "class": observatory_class,
        "bands": list(bands),
        "speclite_fmt": speclite_fmt,
        "sncosmo_fmt": sncosmo_fmt,
        "effective_wavelengths": dict(effective_wavelengths or {}),
    }


# Pre-registered observatories (LSST, Roman, and Euclid)
register_observatory(
    name="LSST",
    observatory_class=LSST,
    bands=LSST_BAND_LIST,
    speclite_fmt=lambda band: f"lsst2023-{band}",
    sncosmo_fmt=lambda band: f"lsst{band}",
)
register_observatory(
    name="Roman",
    observatory_class=Roman,
    bands=ROMAN_BAND_LIST,
    speclite_fmt=lambda band: f"Roman-{band}",
    sncosmo_fmt=lambda band: f"{band}",
    effective_wavelengths={
        "F062": 0.620,
        "F087": 0.870,
        "F106": 1.060,
        "F129": 1.290,
        "F146": 1.460,
        "F158": 1.580,
        "F184": 1.840,
        "F213": 2.130,
    },
)
register_observatory(
    name="Euclid",
    observatory_class=Euclid,
    bands=EUCLID_BAND_LIST,
    speclite_fmt=lambda band: f"Euclid-{band}",
    sncosmo_fmt=lambda band: f"euclid{band}",
)


def _get_observatory_name_for_band(band):
    """Return the observatory name that owns *band*, searching the registry.

    :param band: imaging band name
    :type band: str
    :raises ValueError: if no registered observatory claims the band
    """
    for obs_name, info in _OBSERVATORY_REGISTRY.items():
        if band in info["bands"]:
            return obs_name
    raise ValueError(
        f"Band '{band}' is not recognised by any registered observatory. "
        f"Registered bands: { {o: i['bands'] for o, i in _OBSERVATORY_REGISTRY.items()} }"
    )


def get_observatory(band):
    """Determine the observatory based on the imaging band.

    :param band: imaging band name
    :type band: str
    :return: registered observatory name
    :rtype: str
    :raises ValueError: if the band does not belong to any observatory
    """
    return _get_observatory_name_for_band(band)


def get_speclite_filtername(band):
    """Get the speclite filter name corresponding to the given band.

    Supported bands are listed in the module docstring. Additional bands can
    be registered with :func:`register_observatory`.

    :param band: imaging band name
    :type band: str
    :return: speclite filter name
    :rtype: str
    :raises ValueError: if the band is not registered or has no speclite
        filter
    """
    obs_name = get_observatory(band)
    fmt = _OBSERVATORY_REGISTRY[obs_name]["speclite_fmt"]
    if fmt is None:
        raise ValueError(
            f"Observatory '{obs_name}' (band '{band}') has no speclite filter registered."
        )
    return fmt(band)


def get_speclite_filternames(bands):
    """Get speclite filter names for a list of imaging bands.

    :param bands: list of imaging band names, e.g. ``['u', 'F062',
        'VIS']``
    :type bands: list[str]
    :return: speclite filter names in the same order as the input
    :rtype: list[str]
    :raises ValueError: if any band is not recognized or has no speclite
        filter
    """
    return [get_speclite_filtername(band) for band in bands]


def get_sncosmo_filtername(band):
    """Get the sncosmo bandpass name corresponding to the given band.

    :param band: imaging band name
    :type band: str
    :return: sncosmo bandpass name
    :rtype: str
    :raises ValueError: if the band is not registered
    """
    obs_name = get_observatory(band)
    fmt = _OBSERVATORY_REGISTRY[obs_name].get("sncosmo_fmt")

    # If no specific sncosmo_fmt was registered, fallback to the raw band name
    if fmt is None:
        return band

    return fmt(band)


def get_all_supported_bands():
    """Return every band name currently registered across all observatories.

    :return: flat list of band name strings
    :rtype: list[str]
    """
    all_bands = []
    for info in _OBSERVATORY_REGISTRY.values():
        all_bands.extend(info["bands"])
    return all_bands


def get_band_effective_wavelength(band):
    """Return the throughput-weighted effective wavelength of a band.

    The registered speclite response is used when available. Bands that
    are not shipped by speclite use explicitly configured fallback
    values. Band list position is deliberately not used as a wavelength
    proxy.

    :param band: imaging band name
    :type band: str
    :return: effective wavelength in microns
    :rtype: float
    :raises ValueError: if the band is not registered
    """
    if band in _ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON:
        return _ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON[band]

    obs_name = get_observatory(band)
    configured_wavelengths = _OBSERVATORY_REGISTRY[obs_name]["effective_wavelengths"]
    if band in configured_wavelengths:
        return float(configured_wavelengths[band])

    filter_name = get_speclite_filtername(band)
    response = speclite.filters.load_filter(filter_name)
    return float(response.effective_wavelength.to("micron").value)


def get_band_central_wavelength(band):
    """Backward-compatible alias for :func:`get_band_effective_wavelength`."""
    return get_band_effective_wavelength(band)


def get_band_log_wavelength_ratio(band, reference_band):
    """Return ``log(lambda_band / lambda_reference)``.

    Both wavelengths are throughput-weighted effective wavelengths. This
    is the chromatic coordinate used by the local power-law SED
    approximation.
    """
    wavelength = get_band_effective_wavelength(band)
    reference_wavelength = get_band_effective_wavelength(reference_band)
    if wavelength <= 0 or reference_wavelength <= 0:
        raise ValueError("Band effective wavelengths must be positive.")
    return float(np.log(wavelength / reference_wavelength))


def _configure_roman_filters_if_needed(filter_name):
    """Register Roman filters on first use."""
    if filter_name.startswith("Roman-"):
        from slsim.Pipelines.roman_speclite import configure_roman_filters

        configure_roman_filters()


def get_speclite_filter(band):
    """Return the speclite response for a SLSim band or speclite filter name.

    Supported SLSim bands are resolved through the observatory registry.
    If a filter is not yet available in speclite, load it first with
    ``speclite.filters.load_filter`` or register it with
    :func:`register_observatory`.

    :param band: SLSim band name (e.g. ``"i"``, ``"F062"``, ``"VIS"``) or
        speclite filter name (e.g. ``"lsst2023-i"``).
    :type band: str
    :return: speclite filter response
    :rtype: speclite.filters.FilterResponse
    :raises ValueError: if the filter is not available in speclite
    """
    if not isinstance(band, str):
        raise ValueError(f"Expected a string band name, got {type(band).__name__}.")

    try:
        filter_name = get_speclite_filtername(band)
    except ValueError:
        filter_name = band

    _configure_roman_filters_if_needed(filter_name)
    try:
        return speclite.filters.load_filter(filter_name)
    except ValueError as exc:
        supported = ", ".join(get_all_supported_bands())
        raise ValueError(
            f"Band '{band}' is not a registered SLSim band and is not "
            f"available in speclite. Supported SLSim bands: {supported}. "
            "Load the filter first with "
            "speclite.filters.load_filter(...) or register the observatory "
            "with slsim.Util.bandpass.register_observatory()."
        ) from exc


def get_sncosmo_bandpass(band):
    """Return the sncosmo bandpass for a SLSim band or sncosmo name.

    Supported SLSim bands are resolved through the observatory registry, so
    ``"i"``, ``"F062"`` and ``"VIS"`` map to the same filters used by the
    image-simulation module. Physical or already registered sncosmo names such
    as ``"bessellb"`` are passed through unchanged.

    :param band: SLSim band name or sncosmo band name.
    :type band: str
    :return: sncosmo bandpass
    :rtype: sncosmo.Bandpass
    :raises ValueError: if the band cannot be resolved
    """
    try:
        sncosmo_name = get_sncosmo_filtername(band)
    except ValueError:
        sncosmo_name = band

    try:
        return sncosmo.get_bandpass(sncosmo_name)
    except Exception:
        pass

    try:
        filter_response = get_speclite_filter(band)
    except ValueError as exc:
        raise ValueError(f"Band '{band}' is not recognized by sncosmo.") from exc

    bandpass = sncosmo.Bandpass(
        filter_response.wavelength,
        filter_response.response,
        wave_unit="Angstrom",
        name=sncosmo_name,
    )
    sncosmo.register(bandpass, name=sncosmo_name, force=True)
    return bandpass
