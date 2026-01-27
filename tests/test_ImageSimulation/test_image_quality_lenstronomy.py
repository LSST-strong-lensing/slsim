import pytest
from slsim.ImageSimulation.image_quality_lenstronomy import (
    kwargs_single_band,
    get_observatory,
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
        roman_bands = ["F062", "F087", "F106", "F129", "F158", "F184", "F146"]
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


if __name__ == "__main__":
    pytest.main()
