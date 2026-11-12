from slsim.Util.bandpass import get_sncosmo_bandpass, get_speclite_filter


def test_get_sncosmo_bandpass_resolves_survey_bands():
    assert get_sncosmo_bandpass("i").name == "lssti"
    assert get_sncosmo_bandpass("F062").name.lower().startswith("f062")
    assert get_sncosmo_bandpass("VIS").name.lower() == "euclidvis"


def test_get_sncosmo_bandpass_passes_through_existing_bands():
    bandpass = get_sncosmo_bandpass("bessellb")
    assert bandpass.name == "bessellb"
    assert get_sncosmo_bandpass(bandpass) is bandpass


def test_get_speclite_filter_resolves_bands():
    assert get_speclite_filter("i").name == "lsst2023-i"
    assert get_speclite_filter("F062").name == "Roman-F062"
    filter_response = get_speclite_filter("i")
    assert get_speclite_filter(filter_response) is filter_response
