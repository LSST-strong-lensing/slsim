import pytest
from slsim.ImageSimulation.image_quality_lenstronomy import (
    get_observatory,
    get_speclite_filtername,
    get_speclite_filternames,
)


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
        invalid_bands = ["X", "invalid", "F999", ""]
        for band in invalid_bands:
            with pytest.raises(ValueError, match=f"Band {band} not recognized"):
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
        invalid_bands = ["X", "invalid", "F999", ""]
        for band in invalid_bands:
            with pytest.raises(ValueError, match=f"Band {band} not recognized"):
                get_speclite_filtername(band)

    def test_get_speclite_filternames(self):
        """Test that get_speclite_filternames returns correct list of filter
        names."""
        bands = ["u", "F106", "VIS"]
        expected = ["lsst2023-u", "Roman-F106", "Euclid-VIS"]

        assert get_speclite_filternames(bands) == expected
