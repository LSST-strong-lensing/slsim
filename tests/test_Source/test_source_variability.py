import numpy as np
from slsim.Sources.source_variability.variability import Variability
from slsim.Sources.source_variability.variability import interpolate_variability
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

    def test_interpolation_for_sinusoidal(self):
        kwargs_model00 = {"amp": 1.0, "freq": 0.5}
        kwargs_model01 = {"amp": 2.0, "freq": 1}
        obs_snapshots = np.array([0, 1, np.pi])
        new_snapshots = np.array([0, 0.5, 1, 1.5])
        movie = np.zeros((3, 2, 2))
        movie[:, 0, 0] = Variability(
            "sinusoidal", **kwargs_model00
        ).variability_at_time(obs_snapshots)
        movie[:, 0, 1] = Variability(
            "sinusoidal", **kwargs_model01
        ).variability_at_time(obs_snapshots)
        movie[:, 1, 0] = np.array([4, 2, 2])
        movie[:, 1, 1] = np.array([0, -6, 0])
        interp_movie = interpolate_variability(movie, obs_snapshots, new_snapshots)
        expect_movie = np.zeros((4, 2, 2))
        expect_movie[:, 0, 0] = np.array(
            [0.0, 0.0, 0.0, (np.sin(np.pi * np.pi) - 0.0) * 0.5 / (np.pi - 1.0)]
        )
        expect_movie[:, 0, 1] = np.array(
            [0.0, 0.0, 0.0, (2 * np.sin(2 * np.pi * np.pi) - 0.0) * 0.5 / (np.pi - 1.0)]
        )
        expect_movie[:, 1, 0] = np.array([4.0, 3.0, 2.0, 2.0])
        expect_movie[:, 1, 1] = np.array(
            [0.0, -3.0, -6.0, -6 + (0.0 + 6.0) * 0.5 / (np.pi - 1.0)]
        )
        npt.assert_almost_equal(interp_movie, expect_movie, decimal=5)


if __name__ == "__main__":
    pytest.main()
