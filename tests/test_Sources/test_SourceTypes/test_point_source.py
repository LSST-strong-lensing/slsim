from slsim.Sources.SourceTypes.point_source import PointSource
import numpy as np
import pytest
from astropy import cosmology


class TestPointSource:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

        self.source_dict_sn = {
            "z": 1.0,
            "ps_mag_i": 20,
            "center_x": 0.044,
            "center_y": -0.05,
        }
        kwargs_sn = {
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i", "r"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None,
        }
        self.source_sn = PointSource(
            source_type="supernova", cosmo=cosmo, **kwargs_sn, **self.source_dict_sn
        )

        source_dict_quasar = {"z": 0.8, "ps_mag_i": 20}
        variable_agn_kwarg_dict = {
            "length_of_light_curve": 500,
            "time_resolution": 1,
            "log_breakpoint_frequency": 1 / 20,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "standard_deviation": 0.9,
        }
        kwargs_quasar = {
            "variability_model": "light_curve",
            "kwargs_variability": {"agn_lightcurve", "i", "r"},
            "agn_driving_variability_model": "bending_power_law",
            "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
            "lightcurve_time": np.linspace(0, 1000, 1000),
        }
        self.source_quasar = PointSource(
            source_type="quasar", cosmo=cosmo, **kwargs_quasar, **source_dict_quasar
        )

        source_dict_general_lc = {
            "z": 0.8,
            "MJD": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "ps_mag_i": np.array([15, 16, 17, 18, 19, 20, 21, 22, 23]),
        }
        kwargs_general_lc = {
            "variability_model": "light_curve",
        }

        self.source_general_lc = PointSource(
            source_type="general_lightcurve",
            cosmo=cosmo,
            **kwargs_general_lc,
            **source_dict_general_lc
        )

    def test_redshift(self):
        assert self.source_sn.redshift == 1.0

    def test_source_position(self):
        # no host galaxy. So, point and extended source position are the same.
        x_pos_1, y_pos_1 = self.source_sn.point_source_position
        assert x_pos_1 == self.source_dict_sn["center_x"]
        assert y_pos_1 == self.source_dict_sn["center_y"]

    def test_point_source_magnitude(self):
        # supernova is randomly selected. Can't assert a fix value. Just checking that
        # lightcurve is generated.
        assert self.source_sn.point_source_magnitude(band="i") is not None
        assert self.source_quasar.point_source_magnitude(band="i") == 20
        assert (
            self.source_general_lc.point_source_magnitude(
                band="i", image_observation_times=5
            )
            == 19
        )
        expected_result = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23])
        assert np.all(
            self.source_general_lc.point_source_magnitude(band="i")
            == np.mean(expected_result)
        )
        with pytest.raises(ValueError):
            self.source_general_lc.point_source_magnitude(band="g")

    def test_error(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict_sn = {
            "z": 1.0,
            "ps_mag_i": 20,
            "center_x": 0.044,
            "center_y": -0.05,
        }
        kwargs_sn = {
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i", "r"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None,
        }
        with pytest.raises(ValueError):
            PointSource(cosmo=cosmo, source_type="other", **kwargs_sn, **source_dict_sn)
