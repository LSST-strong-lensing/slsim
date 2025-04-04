from slsim.Sources.SourceTypes.source_base import SourceBase
import numpy as np
from numpy import testing as npt
from astropy.table import Table
import pytest


class TestSourceBase:
    def setup_method(self):
        self.source_dict = {
            "z": 0.8,
            "mag_i": 23,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.002,
            "e2": 0.004,
        }
        self.source_dict2 = {
            "z": [0.7],
            "mag_i": [23],
            "n_sersic": [1],
            "angular_size": [0.2],
            "e1": [0.002],
            "e2": [0.004],
        }
        self.source_dict3 = {
            "z": 0.8,
            "mag_i": 23,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.002,
            "e2": 0.004,
            "center_x": 0.035,
            "center_y": 0.044,
            "ra_off": 0.001,
            "dec_off": 0.002,
        }
        self.source_table = Table([self.source_dict])
        self.source = SourceBase(source_dict=self.source_dict)
        self.source2 = SourceBase(source_dict=self.source_dict2)
        self.source3 = SourceBase(source_dict=self.source_dict3)
        self.source4 = SourceBase(source_dict=self.source_table)

    def test_redshift(self):
        assert self.source.redshift == 0.8
        assert self.source2.redshift == 0.7
        assert self.source4.redshift == 0.8

    def test_point_source_offset(self):
        offset = self.source.point_source_offset
        assert offset[0] is None
        assert offset[1] is None

    def test_source_position(self):
        x, y = self.source.extended_source_position(
            reference_position=[0, 0], draw_area=4 * np.pi
        )
        x2, y2 = self.source.point_source_position(
            reference_position=[0, 0], draw_area=4 * np.pi
        )
        x3, y3 = self.source3.extended_source_position(
            reference_position=[0, 0], draw_area=4 * np.pi
        )
        x4, y4 = self.source3.point_source_position(
            reference_position=[0, 0], draw_area=4 * np.pi
        )
        assert -2 < x < 2
        assert -2 < y < 2
        assert -2 < x2 < 2
        assert -2 < y2 < 2
        assert x == x2
        assert y == y2
        assert x3 == 0.035
        assert y3 == 0.044
        npt.assert_almost_equal(x4, 0.036, decimal=4)
        npt.assert_almost_equal(y4, 0.046, decimal=4)

    def test_angular_size(self):
        assert self.source.angular_size is None

    def test_n_sersic(self):
        assert self.source.n_sersic is None

    def ellipticity(self):
        assert self.source.ellipticity is None

    def test_sersicweight(self):
        assert self.source.sersicweight is None

    def test_image_redshift(self):
        assert self.source.image_redshift is None

    def test_image(self):
        assert self.source.image is None

    def test_phi(self):
        assert self.source.phi is None

    def test_pixel_scale(self):
        assert self.source.pixel_scale is None

    def test_light_curve(self):
        assert self.source.light_curve is None


if __name__ == "__main__":
    pytest.main()
