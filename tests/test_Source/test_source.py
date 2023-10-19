from slsim.Sources.source import Source
import numpy as np
import pytest
from numpy import testing as npt
from astropy.table import Table


class TestSource:
    def setup_method(self):
        source_dict = Table(
            [[0.5], [17], [18], [16], [0.5], [2], [4], [0.35], [0.8], [0.76]],
            names=(
                "z",
                "mag_r",
                "mag_g",
                "mag_i",
                "amp",
                "freq",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
            ),
        )
        self.source = Source(
            source_dict,
            variability_model="sinusoidal",
            kwargs_variab={"amp", "freq"},
        )

    def test_redshift(self):
        assert self.source.redshift == [0.5]

    def test_n_sersic(self):
        assert self.source.n_sersic == [4]

    def test_angular_size(self):
        assert self.source.angular_size == [0.35]

    def test_ellipticity(self):
        assert self.source.ellipticity[0] == [0.8]
        assert self.source.ellipticity[1] == [0.76]

    def test_magnitude_no_variability(self):
        result = self.source.magnitude("r")
        assert result == [17]

    def test_magnitude_with_variability(self):
        image_observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = self.source.magnitude("r", image_observation_times)
        result_comp = np.array([17.48917028, 17.38842661, 17.27946793])
        npt.assert_almost_equal(result, result_comp, decimal=5)


if __name__ == "__main__":
    pytest.main()
