import pytest

import slsim.Util.bandpass as bandpass_module
from slsim.ImageSimulation.image_quality_lenstronomy import register_observatory
from slsim.Util.bandpass import get_sncosmo_bandpass, get_speclite_filter


class _TestObservatory:
    def __init__(self, band, **kwargs):
        self.band = band

    def kwargs_single_band(self):
        return {}


def test_get_sncosmo_bandpass_resolves_survey_bands():
    assert get_sncosmo_bandpass("i").name == "lssti"
    assert get_sncosmo_bandpass("F062").name.lower().startswith("f062")
    assert get_sncosmo_bandpass("VIS").name.lower() == "euclidvis"


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
    assert bandpass.name == "TESTBAND"
    assert get_sncosmo_bandpass("slsim_test_band_not_in_sncosmo") is bandpass


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
