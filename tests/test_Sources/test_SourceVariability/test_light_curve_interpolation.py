import numpy as np
from slsim.Sources.SourceVariability.light_curve_interpolation import (
    LightCurveInterpolation,
)
import pytest


class TestLightCurveInterpolation:
    def test_magnitude(self):
        light_curve_test = {
            "MJD": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "ps_mag_i": np.array([20.0, 21.0, 22.0, 23.0, 24.0]),
        }
        self.light_curve = LightCurveInterpolation(light_curve=light_curve_test)
        observation_times = np.array([1.5, 2.5, 3.5])
        expected_magnitudes = np.array([20.5, 21.5, 22.5])
        result = self.light_curve.magnitude(observation_times)
        assert np.all(result) == np.all(expected_magnitudes)


if __name__ == "__main__":
    pytest.main()
