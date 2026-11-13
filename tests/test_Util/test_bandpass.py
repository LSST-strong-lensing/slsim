import pytest
import sncosmo

import slsim.Util.bandpass as bandpass_module
from slsim.Util.bandpass import (
    EUCLID_BAND_LIST,
    LSST_BAND_LIST,
    ROMAN_BAND_LIST,
    get_all_supported_bands,
    get_observatory,
    get_sncosmo_bandpass,
    get_sncosmo_filtername,
    get_speclite_filter,
    get_speclite_filtername,
    register_observatory,
)


@pytest.fixture(autouse=True)
def _restore_bandpass_state():
    original_registry = dict(bandpass_module._OBSERVATORY_REGISTRY)
    original_cache = dict(bandpass_module._SNCOSMO_BANDPASS_CACHE)
    yield
    bandpass_module._OBSERVATORY_REGISTRY.clear()
    bandpass_module._OBSERVATORY_REGISTRY.update(original_registry)
    bandpass_module._SNCOSMO_BANDPASS_CACHE.clear()
    bandpass_module._SNCOSMO_BANDPASS_CACHE.update(original_cache)


class _TestObservatory:
    def __init__(self, band, **kwargs):
        self.band = band

    def kwargs_single_band(self):
        return {}


def test_get_sncosmo_bandpass_resolves_survey_bands():
    assert get_sncosmo_bandpass("i").name == "lssti"
    assert get_sncosmo_bandpass("F062").name.lower().startswith("f062")
    assert get_sncosmo_filtername("VIS") == "euclidVIS"
    assert get_speclite_filter("VIS").name == "Euclid-VIS"
    assert isinstance(get_sncosmo_bandpass("VIS"), sncosmo.Bandpass)


def test_get_sncosmo_bandpass_vis_fallback_uses_sncosmo_name(monkeypatch):
    original_get_bandpass = sncosmo.get_bandpass

    def fake_get_bandpass(name):
        if name == "euclidVIS":
            raise KeyError(name)
        return original_get_bandpass(name)

    monkeypatch.setattr(sncosmo, "get_bandpass", fake_get_bandpass)

    bandpass = get_sncosmo_bandpass("VIS")
    assert bandpass.name == "euclidVIS"


def test_get_sncosmo_bandpass_passes_through_existing_bands():
    bandpass = get_sncosmo_bandpass("bessellb")
    assert bandpass.name == "bessellb"
    assert get_sncosmo_bandpass(bandpass) is bandpass


def test_get_sncosmo_bandpass_rejects_unknown_band():
    with pytest.raises(ValueError, match="not recognized by sncosmo"):
        get_sncosmo_bandpass("not_a_real_band")


def test_get_sncosmo_bandpass_builds_from_speclite():
    register_observatory(
        name="SLSimTests",
        observatory_class=_TestObservatory,
        bands=["TESTBAND"],
        speclite_fmt=lambda band: "lsst2023-i",
        sncosmo_fmt=lambda band: "slsim_test_band_not_in_sncosmo",
    )

    bandpass = get_sncosmo_bandpass("TESTBAND")
    assert bandpass.name == "slsim_test_band_not_in_sncosmo"
    assert get_sncosmo_bandpass("TESTBAND") is bandpass
    assert get_sncosmo_bandpass("slsim_test_band_not_in_sncosmo") is bandpass

    # The fallback must not leak a bare name into sncosmo's global registry.
    with pytest.raises(Exception):
        sncosmo.get_bandpass("TESTBAND")


def test_get_speclite_filter_resolves_bands():
    assert get_speclite_filter("i").name == "lsst2023-i"
    assert get_speclite_filter("F062").name == "Roman-F062"
    assert get_speclite_filter("lsst2023-i").name == "lsst2023-i"
    filter_response = get_speclite_filter("i")
    assert get_speclite_filter(filter_response) is filter_response


def test_get_speclite_filter_retries_after_registering_filters(monkeypatch):
    calls = []
    original_load_filter = bandpass_module.speclite.filters.load_filter

    def fake_load_filter(name):
        calls.append(name)
        if len(calls) == 1:
            raise ValueError("force fallback")
        return original_load_filter("lsst2023-i")

    monkeypatch.setattr(
        bandpass_module.speclite.filters, "load_filter", fake_load_filter
    )
    monkeypatch.setattr(
        bandpass_module,
        "_configure_roman_filters_if_needed",
        lambda name: calls.append("configure"),
    )

    assert get_speclite_filter("F062").name == "lsst2023-i"
    assert calls == ["Roman-F062", "configure", "Roman-F062"]


def test_configure_roman_filters_if_needed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "slsim.Pipelines.roman_speclite.configure_roman_filters",
        lambda: calls.append("configured"),
    )

    bandpass_module._configure_roman_filters_if_needed("Roman-F062")
    bandpass_module._configure_roman_filters_if_needed("lsst2023-i")

    assert calls == ["configured"]


def test_default_band_conventions_are_shared():
    assert get_observatory("i") == "LSST"
    assert get_speclite_filtername("i") == "lsst2023-i"
    assert get_sncosmo_filtername("i") == "lssti"
    assert get_speclite_filtername("F062") == "Roman-F062"
    assert get_speclite_filtername("VIS") == "Euclid-VIS"

    supported = set(get_all_supported_bands())
    assert set(LSST_BAND_LIST + ROMAN_BAND_LIST + EUCLID_BAND_LIST) <= supported


def test_get_speclite_filter_explains_how_to_load_missing_filter():
    with pytest.raises(ValueError, match="Supported SLSim bands"):
        get_speclite_filter("not_a_real_speclite_filter")


def test_get_speclite_filter_rejects_non_string_input():
    with pytest.raises(TypeError, match="Expected a band name"):
        get_speclite_filter(123)


def test_band_effective_wavelength_without_speclite_or_explicit_wavelength():
    bandpass_module.register_observatory(
        name="NoWavelengthObs",
        observatory_class=_TestObservatory,
        bands=["NO_WAVE"],
        speclite_fmt=None,
    )
    with pytest.raises(ValueError, match="neither an explicitly configured"):
        bandpass_module.get_band_effective_wavelength("NO_WAVE")
