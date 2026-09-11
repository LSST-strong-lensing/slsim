from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Util.color_gradient import default_reference_band
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
import numpy as np
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
            "e1_0": 0.1,
            "e1_1": 0.002,
            "e2_0": 0.001,
            "e2_1": 0.003,
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
                e1_slsim=self.source_dict["e1_0"],
                e2_slsim=self.source_dict["e2_0"],
            )
        )

        assert results[0]["e1"] == e1_light_source_1_lenstronomy
        assert results[0]["e2"] == e2_light_source_1_lenstronomy
        npt.assert_almost_equal(results[0]["magnitude"], 23.994, decimal=3)
        assert results[1]["R_sersic"] == 0.15
        e1_light_source_2_lenstronomy, e2_light_source_2_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self.source_dict["e1_1"],
                e2_slsim=self.source_dict["e2_1"],
            )
        )

        assert results[1]["e1"] == e1_light_source_2_lenstronomy
        assert results[1]["e2"] == e2_light_source_2_lenstronomy
        npt.assert_almost_equal(results[1]["magnitude"], 23.554, decimal=3)
        npt.assert_almost_equal(results2[0]["magnitude"], 1.994, decimal=3)
        npt.assert_almost_equal(results2[1]["magnitude"], 1.554, decimal=3)

        assert source_model[0] == "SERSIC_ELLIPSE"
        assert source_model[1] == "SERSIC_ELLIPSE"

    def test_kwargs_extended_light_returns_independent_model_list(self):
        source_model, _ = self.source.kwargs_extended_light(band="i")
        source_model.append("SERSIC")

        assert self.source._light_model_list == [
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
        ]

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 21.313, decimal=3)

    def test_band_dependent_color_gradient(self):
        source_dict = dict(self.source_dict)
        source_dict.update(
            {
                "mag_g": 23,
                "mag_y": 23,
                "color_gradient": {
                    "component_spectral_slopes": [2.0, -1.0],
                    "reference_band": "i",
                },
            }
        )
        source = DoubleSersic(**source_dict)

        _, kwargs_g = source.kwargs_extended_light(band="g")
        _, kwargs_y = source.kwargs_extended_light(band="y")

        flux_g0 = 10 ** (-kwargs_g[0]["magnitude"] / 2.5)
        flux_g1 = 10 ** (-kwargs_g[1]["magnitude"] / 2.5)
        flux_y0 = 10 ** (-kwargs_y[0]["magnitude"] / 2.5)
        flux_y1 = 10 ** (-kwargs_y[1]["magnitude"] / 2.5)

        assert flux_y0 / (flux_y0 + flux_y1) > flux_g0 / (flux_g0 + flux_g1)

    def test_color_gradient_disabled_or_missing_slopes_uses_base_weights(self):
        assert self.source._weights_for_band("i") == (0.4, 0.6)

        source_dict = dict(self.source_dict)
        source_dict["color_gradient"] = {}
        no_gradient_source = DoubleSersic(**source_dict)
        assert no_gradient_source._weights_for_band("i") == (0.4, 0.6)

    def test_default_reference_band_and_weight_validation(self):
        source_dict = dict(self.source_dict)
        source_dict.update(
            {
                "mag_g": 23,
                "mag_y": 23,
                "color_gradient": {"component_spectral_slopes": [1.0, 0.0]},
            }
        )
        source = DoubleSersic(**source_dict)

        assert default_reference_band(source.source_dict) == "i"
        assert source._weights_for_band("i") == (0.4, 0.6)

        source_without_magnitudes = dict(self.source_dict)
        source_without_magnitudes.pop("mag_i")
        source_without_magnitudes["color_gradient"] = {
            "component_spectral_slopes": [1.0, 0.0]
        }
        no_magnitude_source = DoubleSersic(**source_without_magnitudes)
        assert default_reference_band(no_magnitude_source.source_dict) == "i"

        source_dict["color_gradient"] = "invalid"
        with pytest.raises(ValueError, match="must be a dictionary"):
            DoubleSersic(**source_dict)._weights_for_band("i")

        source_dict["color_gradient"] = {
            "component_spectral_slopes": [1.0],
        }
        with pytest.raises(ValueError, match="must match the number of components"):
            DoubleSersic(**source_dict)._weights_for_band("i")

        source_dict["color_gradient"] = {
            "component_spectral_slopes": [1.0, 0.0],
            "min_weight": 0.5,
        }
        with pytest.raises(ValueError, match="must be in"):
            DoubleSersic(**source_dict)._weights_for_band("i")

    def test_color_gradient_clips_component_weight(self):
        source_dict = dict(self.source_dict)
        source_dict["color_gradient"] = {
            "component_spectral_slopes": [100.0, 0.0],
            "reference_band": "i",
            "min_weight": 0.2,
        }
        source = DoubleSersic(**source_dict)

        w0, w1 = source._weights_for_band("F213")
        assert np.isclose(w0, 0.8)
        assert np.isclose(w0 + w1, 1.0)


if __name__ == "__main__":
    pytest.main()
