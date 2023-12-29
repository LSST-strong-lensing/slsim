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
        var=Variability(variability_model="light_curve", MJD=mjd, ps_mag_1=ps_mag_1)
        results = var.variability_at_time(observation_times)
        assert np.all(results) == np.all(ps_mag_1)

    def test_initialization_invalid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        with pytest.raises(ValueError) as excinfo:
            Variability("invalid_model", **kwargs_model)
        assert (
                "Given model is not supported. Currently supported models are" 
                "sinusoidal, light_curve."
            ) in str(excinfo.value)

    def test_variability_at_t_sinusoidal(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = variability.variability_at_time(observation_times)
        expected_result = np.array([0.43030122, 0.97536797, 0.14773276])
        npt.assert_almost_equal(result, expected_result, decimal=5)


if __name__ == "__main__":
    pytest.main()
