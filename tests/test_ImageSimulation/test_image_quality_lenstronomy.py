import pytest
from slsim.ImageSimulation.image_quality_lenstronomy import (
    check_speclite_name,
    get_speclite_filtername,
    get_speclite_filternames,
    kwargs_single_band,
    get_observatory,
    register_observatory,
    get_all_supported_bands,
)


class DummyObservatory:
    """A dummy observatory class for testing the registry."""

    def __init__(self, band, **kwargs):
        self.band = band
        self.kwargs = kwargs

    def kwargs_single_band(self):
        return {"pixel_scale": 0.1, "exposure_time": 100, "dummy_band": self.band}


class TestRegistrySystem:
    """Tests for modular observatory registry system."""

    def test_check_speclite_name(self):
        """Test that check_speclite_name returns the band if it exists in speclite."""
        
        # valid filter
        valid_filter = "sdss2010-r"
        assert check_speclite_name(valid_filter) == valid_filter
        
        # invalid filter
        invalid_filter = "invalid-filter-123"
        assert check_speclite_name(invalid_filter) is None

    def test_register_new_observatory(self):
        """Test registering a custom observatory and retrieving its data."""
        register_observatory(
            name="TestObs",
            observatory_class=DummyObservatory,
            bands=["T1", "T2"],
            speclite_fmt=lambda b: f"TestObs-{b}",
        )

        # Test observatory retrieval
        assert get_observatory("T1") == "TestObs"

        # Test speclite name retrieval
        assert get_speclite_filtername("T2") == "TestObs-T2"

        # Test kwargs_single_band auto-lookup
        kwargs = kwargs_single_band("T1")
        assert kwargs["dummy_band"] == "T1"
        assert kwargs["pixel_scale"] == 0.1

    def test_register_observatory_no_speclite(self):
        """Test registering an observatory without speclite format."""
        register_observatory(
            name="NoSpecObs",
            observatory_class=DummyObservatory,
            bands=["N1"],
            speclite_fmt=None,
        )

        with pytest.raises(ValueError, match="has no speclite filter registered"):
            get_speclite_filtername("N1")

    def test_register_observatory_default_speclite(self):
        """Test registering an observatory using the default check_speclite_name format."""
        # Test with a valid speclite filter
        register_observatory(
            name="DefaultObsValid",
            observatory_class=DummyObservatory,
            bands=["sdss2010-r"],
        )
        assert get_speclite_filtername("sdss2010-r") == "sdss2010-r"

        # Test with an invalid speclite filter
        register_observatory(
            name="DefaultObsInvalid",
            observatory_class=DummyObservatory,
            bands=["invalid-filter-123"],
        )
        # check_speclite_name returns None for invalid bands
        assert get_speclite_filtername("invalid-filter-123") is None


class TestGetObservatory:
    """Tests for the get_observatory function."""

    def test_lsst_bands(self):
        """Test that LSST bands return 'LSST'."""
        lsst_bands = ["u", "g", "r", "i", "z", "y"]
        for band in lsst_bands:
            assert get_observatory(band) == "LSST"

    def test_roman_bands(self):
        """Test that Roman bands return 'Roman'."""
        roman_bands = ["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"]
        for band in roman_bands:
            assert get_observatory(band) == "Roman"

    def test_euclid_bands(self):
        """Test that Euclid bands return 'Euclid'."""
        euclid_bands = ["VIS"]
        for band in euclid_bands:
            assert get_observatory(band) == "Euclid"

    def test_invalid_band_raises_error(self):
        """Test that invalid bands raise a ValueError."""
        invalid_bands = ["X", "invalid", "F999"]
        for band in invalid_bands:
            with pytest.raises(ValueError, match=f"Band '{band}' is not recognised"):
                get_observatory(band)


class TestGetSpecliteFiltername:
    """Tests for the get_speclite_filtername function."""

    def test_lsst_bands(self):
        """Test that LSST bands return correct speclite filter names."""
        lsst_bands = ["u", "g", "r", "i", "z", "y"]
        for band in lsst_bands:
            expected = f"lsst2023-{band}"
            assert get_speclite_filtername(band) == expected

    def test_roman_bands(self):
        """Test that Roman bands return correct speclite filter names."""
        roman_bands = ["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"]
        for band in roman_bands:
            expected = f"Roman-{band}"
            assert get_speclite_filtername(band) == expected

    def test_euclid_bands(self):
        """Test that Euclid bands return correct speclite filter names."""
        euclid_bands = ["VIS"]
        for band in euclid_bands:
            expected = f"Euclid-{band}"
            assert get_speclite_filtername(band) == expected

    def test_invalid_band_raises_error(self):
        """Test that invalid bands raise a ValueError."""
        invalid_bands = ["X", "invalid", "F999"]
        for band in invalid_bands:
            with pytest.raises(ValueError, match=f"Band '{band}' is not recognised"):
                get_speclite_filtername(band)

    def test_get_speclite_filternames(self):
        """Test that get_speclite_filternames returns correct list of filter
        names."""
        bands = ["u", "F106", "VIS"]
        expected = ["lsst2023-u", "Roman-F106", "Euclid-VIS"]
        assert get_speclite_filternames(bands) == expected


class TestKwargsSingleBand:
    """Tests for the kwargs_single_band function."""

    def test_lsst_band_returns_dict(self):
        """Test that LSST band returns a dictionary with expected keys."""
        result = kwargs_single_band(band="i", observatory="LSST")
        assert isinstance(result, dict)
        assert "pixel_scale" in result
        assert "exposure_time" in result

    def test_roman_band_returns_dict(self):
        """Test that Roman band returns a dictionary with expected keys."""
        result = kwargs_single_band(band="F106", observatory="Roman")
        assert isinstance(result, dict)
        assert "pixel_scale" in result
        assert "exposure_time" in result

    def test_euclid_band_returns_dict(self):
        """Test that Euclid band returns a dictionary with expected keys."""
        result = kwargs_single_band(band="VIS", observatory="Euclid")
        assert isinstance(result, dict)
        assert "pixel_scale" in result
        assert "exposure_time" in result

    def test_auto_observatory_lookup(self):
        """Test that kwargs_single_band automatically finds the observatory if
        None."""
        result = kwargs_single_band(band="i")  # Should auto-resolve to LSST
        assert isinstance(result, dict)
        assert "pixel_scale" in result
        assert "exposure_time" in result

    def test_unregistered_observatory_raises_error(self):
        """Test that providing an explicitly unregistered observatory raises a
        ValueError."""
        with pytest.raises(ValueError, match="is not registered"):
            kwargs_single_band(band="i", observatory="FakeObs")


def test_get_all_supported_bands_contains_defaults():
    """Test that all default bands are present in the returned list."""
    all_bands = get_all_supported_bands()
    expected_bands = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "F062",
        "F087",
        "F106",
        "F129",
        "F158",
        "F184",
        "F146",
        "F213",
        "VIS",
    ]
    for band in expected_bands:
        assert band in all_bands


if __name__ == "__main__":
    pytest.main()
