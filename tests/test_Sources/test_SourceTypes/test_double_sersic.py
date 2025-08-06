from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
import pytest
from numpy import testing as npt


class TestDoubleSersic:
    def setup_method(self):
        self.source_dict = {
            "z": 0.5,
            "n_sersic_0": 1,
            "n_sersic_1": 4,
            "angular_size_0": 0.2,
            "angular_size_1": 0.15,
            "e1_1": 0.1,
            "e1_2": 0.002,
            "e2_1": 0.001,
            "e2_2": 0.003,
            "w0": 0.4,
            "w1": 0.6,
            "mag_i": 23,
        }
        self.source = DoubleSersic(**self.source_dict)

    def test_angular_size(self):
        assert self.source._angular_size_list[0] == 0.2
        assert self.source._angular_size_list[1] == 0.15
        npt.assert_almost_equal(self.source.angular_size, 0.2, decimal=1)

    def test_sersicweight(self):
        w0, w1 = self.source._w0, self.source._w1
        assert w0 == 0.4
        assert w1 == 0.6

    def test_ellipticity(self):
        e1, e2 = self.source.ellipticity
        npt.assert_almost_equal(e1, -0.017, decimal=3)
        npt.assert_almost_equal(e2, 0.002, decimal=3)

    def test_n_sersic(self):
        assert self.source._n_sersic[0] == 1
        assert self.source._n_sersic[1] == 4

    def test_extended_source_magnitude(self):
        assert self.source.extended_source_magnitude("i") == 23
        with pytest.raises(ValueError):
            self.source.extended_source_magnitude("g")

    def test_kwargs_extended_source_light(self):
        source_model, results = self.source.kwargs_extended_light(band="i")
        _, results2 = self.source.kwargs_extended_light(band=None)
        assert results[0]["R_sersic"] == 0.2

        e1_light_source_1_lenstronomy, e2_light_source_1_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self.source_dict["e1_1"],
                e2_slsim=self.source_dict["e2_1"],
            )
        )

        assert results[0]["e1"] == e1_light_source_1_lenstronomy
        assert results[0]["e2"] == e2_light_source_1_lenstronomy
        npt.assert_almost_equal(results[0]["magnitude"], 23.994, decimal=3)
        assert results[1]["R_sersic"] == 0.15
        e1_light_source_2_lenstronomy, e2_light_source_2_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self.source_dict["e1_2"],
                e2_slsim=self.source_dict["e2_2"],
            )
        )

        assert results[1]["e1"] == e1_light_source_2_lenstronomy
        assert results[1]["e2"] == e2_light_source_2_lenstronomy
        npt.assert_almost_equal(results[1]["magnitude"], 23.554, decimal=3)
        npt.assert_almost_equal(results2[0]["magnitude"], 1.994, decimal=3)
        npt.assert_almost_equal(results2[1]["magnitude"], 1.554, decimal=3)

        assert source_model[0] == "SERSIC_ELLIPSE"
        assert source_model[1] == "SERSIC_ELLIPSE"

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 21.313, decimal=3)


if __name__ == "__main__":
    pytest.main()
