from slsim.Sources.SourceTypes.single_sersic import SingleSersic
import pytest
from numpy import testing as npt


class TestSingleSersic:
    def setup_method(self):
        self.source_dict = {
            "z": 0.8,
            "mag_i": 23,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.002,
            "e2": 0.004,
        }
        self.source = SingleSersic(**self.source_dict)

    def test_angular_size(self):
        assert self.source.angular_size == 0.2

    def test_ellipticity(self):
        e1, e2 = self.source.ellipticity
        assert e1 == 0.002
        assert e2 == 0.004

    def test_n_sersic(self):
        assert self.source._n_sersic == 1

    def test_extended_source_magnitude(self):
        assert self.source.extended_source_magnitude("i") == 23
        with pytest.raises(ValueError):
            self.source.extended_source_magnitude("g")

    def test_kwargs_extended_source_light(self):
        source_model, results = self.source.kwargs_extended_light(band="i")
        _, results2 = self.source.kwargs_extended_light(band=None)
        assert results[0]["R_sersic"] == 0.2
        assert results[0]["e1"] == -0.002
        assert results[0]["e2"] == 0.004
        assert results[0]["magnitude"] == 23
        assert results2[0]["magnitude"] == 1

        assert source_model[0] == "SERSIC_ELLIPSE"

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 21.500, decimal=3)


if __name__ == "__main__":
    pytest.main()
