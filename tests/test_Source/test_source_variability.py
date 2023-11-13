import numpy as np
from slsim.Sources.source_variability.variability import Variability
from numpy import testing as npt
import pytest


class TestVariability:
    def test_initialization_valid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        assert variability.variability_model == "sinusoidal"

    def test_initialization_invalid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        with pytest.raises(ValueError) as excinfo:
            Variability("invalid_model", **kwargs_model)
        assert (
            "Given model is not supported. Currently supported model is sinusoidal."
        ) in str(excinfo.value)

    def test_variability_at_t_sinusoidal(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = variability.variability_at_time(observation_times)
        expected_result = np.array([-0.43030122, -0.97536797, -0.14773276])
        npt.assert_almost_equal(result, expected_result, decimal=5)


if __name__ == "__main__":
    pytest.main()
