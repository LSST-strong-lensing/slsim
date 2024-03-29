from slsim.Sources.source import Source
import numpy as np
import pytest
from numpy import testing as npt
from astropy.table import Table


class TestSource:
    def setup_method(self):
        source_dict = Table(
            [
                [0.5],
                [17],
                [18],
                [16],
                [23],
                [24],
                [22],
                [0.5],
                [2],
                [4],
                [0.35],
                [0.8],
                [0.76],
            ],
            names=(
                "z",
                "ps_mag_r",
                "ps_mag_g",
                "ps_mag_i",
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
        source_dict2 = Table(
            [
                [0.5],
                [np.array([17, 18, 19, 20, 21])],
                [18],
                [16],
                [23],
                [24],
                [22],
                [0.5],
                [2],
                [4],
                [0.35],
                [0.8],
                [0.76],
                [0.001],
                [-0.001],
            ],
            names=(
                "z",
                "ps_mag_r",
                "ps_mag_g",
                "ps_mag_i",
                "mag_r",
                "mag_g",
                "mag_i",
                "amp",
                "freq",
                "n_sersic",
                "angular_size",
                "e1",
                "e2",
                "ra_off",
                "dec_off",
            ),
        )
        self.source = Source(
            source_dict,
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
        )
        self.source2 = Source(
            source_dict2,
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
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

    def test_ps_magnitude_no_variability(self):
        result = self.source.point_source_magnitude("r")
        assert result == [17]

    def test_ps_magnitude_with_variability(self):
        image_observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = self.source.point_source_magnitude("r", image_observation_times)
        result_comp = np.array([0.48917028, 0.38842661, 0.27946793])
        npt.assert_almost_equal(result, result_comp, decimal=5)

    def test_es_magnitude(self):
        result = self.source.extended_source_magnitude("r")
        assert result == [23]

    def test_ps_magnitude_array(self):
        result = self.source2.point_source_magnitude("r")
        assert len(result) == 5

    def test_extended_source_position(self):

        pos = self.source.extended_source_position(
            center_lens=np.array([0.002, -0.002]), draw_area=4 * np.pi
        )
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)

    def test_point_source_position_without_offset(self):
        pos = self.source.point_source_position(
            center_lens=np.array([0.002, -0.002]), draw_area=4 * np.pi
        )
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)

    def test_point_source_position_with_offset(self):
        pos = self.source2.point_source_position(
            center_lens=np.array([0.002, -0.002]), draw_area=4 * np.pi
        )
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)


if __name__ == "__main__":
    pytest.main()
