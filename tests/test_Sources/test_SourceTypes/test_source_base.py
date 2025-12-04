from slsim.Sources.SourceTypes.source_base import SourceBase
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
        self.source = SourceBase(**self.source_dict)
        self.source2 = SourceBase(z=1)
        self.source3 = SourceBase(**self.source_dict3)
        self.source4 = SourceBase(**self.source_table)

    def test_redshift(self):
        assert self.source.redshift == 0.8
        assert self.source4.redshift == 0.8

    def test_point_source_offset(self):
        offset = self.source.point_source_offset
        assert offset[0] == 0
        assert offset[1] == 0

    def test_source_position(self):
        x, y = self.source.extended_source_position
        x2, y2 = self.source.point_source_position
        x3, y3 = self.source3.extended_source_position
        x4, y4 = self.source3.point_source_position
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
        assert self.source.angular_size == 0

        def angular_size_raise():
            source = SourceBase(z=1, extended_source=True)
            return source.angular_size

        npt.assert_raises(ValueError, angular_size_raise)

    def test_ellipticity(self):
        assert self.source.ellipticity == (0, 0)

        def ellipticity_raise():
            source = SourceBase(z=1, extended_source=True)
            return source.ellipticity

        npt.assert_raises(ValueError, ellipticity_raise)

    def test_extended_source_magnitude(self):
        source = SourceBase(z=1, extended_source=False)
        mag = source.extended_source_magnitude(band="i")
        assert mag is None

    def test_kwargs_point_source(self):
        source = SourceBase(z=1, point_source=False)
        source_model, kwargs_source = source.kwargs_point_source(
            band="r", image_observation_times=None, image_pos_x=None, image_pos_y=None
        )
        assert source_model is None
        assert kwargs_source == []
        source = SourceBase(
            z=1, point_source=True, lensed=True, variability_model="NONE"
        )
        npt.assert_raises(
            ValueError,
            source.kwargs_point_source,
            band="r",
            image_observation_times=None,
            image_pos_x=[1],
            image_pos_y=[0],
        )

        npt.assert_raises(
            ValueError,
            source.kwargs_point_source,
            band="r",
            image_observation_times=None,
            ps_mag=[1, 1],
            image_pos_x=[1],
            image_pos_y=[0],
        )

        source = SourceBase(
            z=1, point_source=True, lensed=False, variability_model="NONE", ps_mag_r=20
        )
        source_model, kwargs_source = source.kwargs_point_source(
            band="r", image_observation_times=None, image_pos_x=None, image_pos_y=None
        )

        source_model, kwargs_source_new = source.kwargs_point_source(
            band="r", image_observation_times=None, image_pos_x=None, image_pos_y=None
        )

        assert source_model == "UNLENSED"
        assert kwargs_source["ra_image"] == kwargs_source_new["ra_image"]

        source = SourceBase(
            z=1, point_source=True, lensed=True, variability_model="NONE", ps_mag_r=20
        )
        source_model, kwargs_source = source.kwargs_point_source(
            band="r", image_observation_times=None, image_pos_x=None, image_pos_y=None
        )
        assert source_model == "SOURCE_POSITION"

        npt.assert_raises(
            ValueError,
            source.kwargs_point_source,
            band="r",
            image_observation_times=[1, 2],
            image_pos_x=[1],
            image_pos_y=[0],
        )

    def test_update_microlensing_kwargs_source_morphology(self):
        # Test default pass-through behavior
        source = SourceBase(z=1)
        initial_kwargs = {"param1": 10, "param2": "test"}
        updated_kwargs = source.update_microlensing_kwargs_source_morphology(
            initial_kwargs
        )
        assert updated_kwargs == initial_kwargs


if __name__ == "__main__":
    pytest.main()
