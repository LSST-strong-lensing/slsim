"""Resolve SLSim band names to speclite and sncosmo bandpasses."""

import sncosmo
import speclite.filters

from slsim.ImageSimulation.image_quality_lenstronomy import (
    get_sncosmo_filtername,
    get_speclite_filtername,
)


def _configure_roman_filters_if_needed(filter_name):
    """Register Roman filters on first use."""
    if filter_name.startswith("Roman-"):
        from slsim.Pipelines.roman_speclite import configure_roman_filters

        configure_roman_filters()


def get_speclite_filter(band):
    """Return the speclite response for a SLSim band or speclite filter name.

    :param band: SLSim band name (e.g. ``"i"``, ``"F062"``, ``"VIS"``),
        speclite filter name (e.g. ``"lsst2023-i"``) or an existing
        ``speclite.filters.FilterResponse``.
    :type band: str or speclite.filters.FilterResponse
    :return: speclite filter response
    :rtype: speclite.filters.FilterResponse
    :raises ValueError: if the filter is not registered.
    """
    if isinstance(band, speclite.filters.FilterResponse):
        return band
    try:
        filter_name = get_speclite_filtername(band)
    except ValueError:
        filter_name = band
    try:
        return speclite.filters.load_filter(filter_name)
    except ValueError:
        _configure_roman_filters_if_needed(filter_name)
        return speclite.filters.load_filter(filter_name)


def get_sncosmo_bandpass(band):
    """Return the sncosmo bandpass for a SLSim band or sncosmo name.

    Supported SLSim bands are resolved through the observatory registry, so
    ``"i"``, ``"F062"`` and ``"VIS"`` map to the same filters used by the
    image-simulation module. Physical or already registered sncosmo names such
    as ``"bessellb"`` are passed through unchanged.

    :param band: SLSim band name, sncosmo band name or an existing
        ``sncosmo.Bandpass``.
    :type band: str or sncosmo.Bandpass
    :return: sncosmo bandpass
    :rtype: sncosmo.Bandpass
    :raises ValueError: if the band cannot be resolved.
    """
    if isinstance(band, sncosmo.Bandpass):
        return band
    try:
        sncosmo_name = get_sncosmo_filtername(band)
    except ValueError:
        try:
            return sncosmo.get_bandpass(band)
        except Exception as exc:
            raise ValueError(f"Band '{band}' is not recognized by sncosmo.") from exc
    try:
        return sncosmo.get_bandpass(sncosmo_name)
    except Exception:
        filter_response = get_speclite_filter(band)
        bandpass = sncosmo.Bandpass(
            filter_response.wavelength,
            filter_response.response,
            wave_unit="Angstrom",
            name=band,
        )
        sncosmo.register(bandpass, name=band, force=True)
        if sncosmo_name != band:
            sncosmo.register(bandpass, name=sncosmo_name, force=True)
        return bandpass
