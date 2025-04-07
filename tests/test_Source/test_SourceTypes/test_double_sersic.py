from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
import numpy as np
import pytest
from numpy import testing as npt


class TestDoubleSersic:
    def setup_method(self):
        self.source_dict = {
            "z": 0.5,
            "n_sersic_0": 1,
            "n_sersic_1": 4,
            "angular_size0": 0.2,
            "angular_size1": 0.15,
            "e0_1": 0.001,
            "e0_2": 0.002,
            "e1_1": 0.001,
            "e1_2": 0.003,
            "w0": 0.4,
            "w1": 0.6,
            "mag_i": 23,
        }
        self.source = DoubleSersic(source_dict=self.source_dict)

    def test_angular_size(self):
        assert self.source._angular_size[0] == 0.2
        assert self.source._angular_size[1] == 0.15

    def test_sersicweight(self):
        w0, w1 = self.source._sersicweight
        assert w0 == 0.4
        assert w1 == 0.6

    def test_ellipticity(self):
        e01, e02, e11, e12 = self.source._ellipticity
        assert e01 == 0.001
        assert e02 == 0.002
        assert e11 == 0.001
        assert e12 == 0.003

    def test_n_sersic(self):
        assert self.source.n_sersic[0] == 1
        assert self.source.n_sersic[1] == 4

    def test_extended_source_magnitude(self):
        assert self.source.extended_source_magnitude("i") == 23
        with pytest.raises(ValueError):
            self.source.extended_source_magnitude("g")

    def test_kwargs_extended_source_light(self):
        results = self.source.kwargs_extended_source_light(
            reference_position=[0, 0], draw_area=4 * np.pi, band="i"
        )
        results2 = self.source.kwargs_extended_source_light(
            reference_position=[0, 0], draw_area=4 * np.pi, band=None
        )
        assert results[0]["R_sersic"] == 0.2
        assert results[0]["e1"] == -0.001
        assert results[0]["e2"] == 0.002
        npt.assert_almost_equal(results[0]["magnitude"], 23.994, decimal=3)
        assert results[1]["R_sersic"] == 0.15
        assert results[1]["e1"] == -0.001
        assert results[1]["e2"] == 0.003
        npt.assert_almost_equal(results[1]["magnitude"], 23.554, decimal=3)
        npt.assert_almost_equal(results2[0]["magnitude"], 1.994, decimal=3)
        npt.assert_almost_equal(results2[1]["magnitude"], 1.554, decimal=3)

    def test_extended_source_light_model(self):
        source_model = self.source.extended_source_light_model()
        assert source_model[0] == "SERSIC_ELLIPSE"
        assert source_model[1] == "SERSIC_ELLIPSE"

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 21.210, decimal=3)


if __name__ == "__main__":
    pytest.main()
