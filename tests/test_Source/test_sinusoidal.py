import numpy as np
from slsim.Sources.SourceVariability.sinusoidal_variability import SinusoidalVariability
import pytest


class TestSinusoidalVariability:
    def test_sinusoidal_variability(self):
        amp = 1.0
        freq = 0.1
        sinusoidal_variability = SinusoidalVariability(amp=amp, freq=freq)
        observation_times = np.array([0.0, 1.0, 2.0, 3.0])
        expected_magnitudes = amp * abs(np.sin(2 * np.pi * freq * observation_times))
        result = sinusoidal_variability.magnitude(observation_times)
        assert np.all(result) == np.all(expected_magnitudes)
        assert sinusoidal_variability.amp == 1
        assert sinusoidal_variability.freq == 0.1


if __name__ == "__main__":
    pytest.main()
