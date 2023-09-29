import astropy.units as u
import numpy as np
from sim_pipeline.Sources.source_variability.variability import Variability
import pytest


class TestVariability(object):
    def test_sinudoidal_variability(self):
        variability_params = {"amp": 2.0, "freq": 0.5}
        variability = Variability(**variability_params)
        time = 3 * u.day
        result = variability.sinusoidal_variability(time)
        expected_result = 2.0 * np.sin(2 * np.pi * 0.5 * 3.0)
        time1 = 300 * u.second
        result1 = variability.sinusoidal_variability(time1)
        expected_result1 = 2.0 * np.sin(2 * np.pi * 0.5 * 300.0 / (24 * 3600))
        assert result == expected_result
        assert result1 == expected_result1


if __name__ == "__main__":
    pytest.main()
