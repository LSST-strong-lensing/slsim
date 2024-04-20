import numpy as np
from slsim.Sources.SourceVariability.variability import Variability
from numpy import testing as npt
import pytest


class TestVariability:
    def test_initialization_valid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        assert variability.variability_model == "sinusoidal"

    def test_light_curve_variability(self):
        mjd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ps_mag_1 = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        observation_times = np.array([1, 2, 3, 4, 5])
        var = Variability(variability_model="light_curve", MJD=mjd, ps_mag_1=ps_mag_1)
        results = var.variability_at_time(observation_times)
        assert np.all(results) == np.all(ps_mag_1)

    def test_initialization_invalid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        with pytest.raises(ValueError) as excinfo:
            Variability("invalid_model", **kwargs_model)
        assert (
            "Given model is not supported. Currently supported models are"
            "sinusoidal, light_curve, bending_power_law, user_defined_psd."
        ) in str(excinfo.value)

    def test_variability_at_t_sinusoidal(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = variability.variability_at_time(observation_times)
        expected_result = np.array([0.43030122, 0.97536797, 0.14773276])
        npt.assert_almost_equal(result, expected_result, decimal=5)

    def test_variability_bending_power_law(self):
        kwargs_model = {
            "length_of_light_curve": 1000,
            "time_resolution": 1,
            "log_breakpoint_frequency": -2,
            "mean_magnitude": 0,
            "seed": None,
        }
        var = Variability("bending_power_law", **kwargs_model)
        observation_times = np.linspace(0, 1000, 1000)
        results = var.variability_at_time(observation_times)
        assert var.variability_model == "bending_power_law"
        npt.assert_almost_equal(results.mean(), 0, decimal=5)

    def test_variability_user_defined_psd(self):
        frequencies = np.linspace(1 / 1000, 1 / 2, 1000)
        psd = frequencies ** (-3)
        kwargs_model = {
            "length_of_light_curve": 1000,
            "time_resolution": 1,
            "input_frequencies": frequencies,
            "input_psd": psd,
            "mean_magnitude": 50,
            "seed": None,
        }
        var = Variability("user_defined_psd", **kwargs_model)
        observation_times = np.linspace(0, 999, 1000)
        results = var.variability_at_time(observation_times)
        assert var.variability_model == "user_defined_psd"
        npt.assert_almost_equal(results.mean(), 50)


if __name__ == "__main__":
    pytest.main()
