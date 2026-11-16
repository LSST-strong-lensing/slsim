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
    original_sncosmo = dict(sncosmo.bandpasses._BANDPASSES._instances)
    yield
    bandpass_module._OBSERVATORY_REGISTRY.clear()
    bandpass_module._OBSERVATORY_REGISTRY.update(original_registry)
    sncosmo.bandpasses._BANDPASSES._instances.clear()
    sncosmo.bandpasses._BANDPASSES._instances.update(original_sncosmo)


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


def test_get_sncosmo_bandpass_passes_through_existing_bands():
    bandpass = get_sncosmo_bandpass("bessellb")
    assert bandpass.name == "bessellb"


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
    assert get_sncosmo_bandpass("slsim_test_band_not_in_sncosmo") is bandpass

    # The fallback must not leak a bare name into sncosmo's global registry.
    with pytest.raises(Exception):
        sncosmo.get_bandpass("TESTBAND")


def test_get_speclite_filter_resolves_bands():
    assert get_speclite_filter("i").name == "lsst2023-i"
    assert get_speclite_filter("F062").name == "Roman-F062"
    assert get_speclite_filter("lsst2023-i").name == "lsst2023-i"


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
    with pytest.raises(ValueError, match="Load the filter first"):
        get_speclite_filter("not_a_real_speclite_filter")
