from sim_pipeline.Sources.source import Source
import numpy as np
import astropy.units as u
import pytest
from numpy import testing as npt


class TestSource:
    def setup_method(self):
        self.source_dict = {"z": 0.5, "mag_r": 20.0, "mag_g": 18.0, "mag_i": 21.0}

        self.kwargs_variab = {"variability_model": "sinusoidal", "amp": 2, "freq": 5}

    def test_magnitude_no_variability(self):
        source = Source(self.source_dict)
        mag = source.magnitude("i")
        assert mag == 21.0

    def test_magnitude_with_variability(self):
        source = Source(self.source_dict, self.kwargs_variab)
        mag = source.magnitude(
            "r",
            magnification=np.array([2.0, -0.5]),
            image_observation_times=np.array([[0.0, 1.0], [2, 3]]) * u.day,
        )
        mag_cal = np.array([[19.24742501, 19.24742501], [20.75257499, 20.75257499]])
        npt.assert_almost_equal(mag, mag_cal, decimal=5)

    def test_magnitude_invalid_variability_model(self):
        source = Source(self.source_dict)
        with pytest.raises(ValueError):
            source.magnitude(
                "r",
                magnification=np.array([2.0, -0.5]),
                image_observation_times=np.array([[0.0, 1.0], [2, 3]]) * u.day,
            )

    def test_to_dict(self):
        source = Source(self.source_dict)
        source_dict = source.to_dict()
        assert source_dict == self.source_dict


if __name__ == "__main__":
    pytest.main()
