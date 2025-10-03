from slsim.Sources.SourceTypes.quasar import Quasar, extract_agn_kwargs_from_source_dict
import numpy as np
import numpy.testing as npt
import pytest
from astropy import cosmology


class TestQuasar:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = {"z": 0.8, "ps_mag_i": 20, "random_seed": 42}
        source_dict2 = {
            "z": 0.8,
            "MJD": [0, 2, 3, 4, 5, 6],
            "ps_mag_i": [21, 20, 18, 21, 22, 23],
        }
        source_dict3 = {"z": 0.8}
        variable_agn_kwarg_dict = {
            "length_of_light_curve": 500,
            "time_resolution": 1,
            "log_breakpoint_frequency": 1 / 20,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "standard_deviation": 0.9,
        }
        kwargs_quasar = {
            "source_type": "quasar",
            "variability_model": "light_curve",
            "kwargs_variability": {"agn_lightcurve", "i", "r"},
            "agn_driving_variability_model": "bending_power_law",
            "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
            "lightcurve_time": np.linspace(0, 1000, 1000),
        }

        kwargs_quasar_none = {
            "source_type": "quasar",
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "agn_driving_variability_model": "bending_power_law",
            "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
            "lightcurve_time": np.linspace(0, 1000, 1000),
        }

        self.source = Quasar(cosmo=cosmo, **source_dict, **kwargs_quasar)

        self.source_none = Quasar(cosmo=cosmo, **kwargs_quasar_none, **source_dict)
        self.source_cosmo_error = Quasar(cosmo=None, **kwargs_quasar, **source_dict)
        self.source_light_curve = Quasar(
            cosmo=cosmo, **kwargs_quasar_none, **source_dict2
        )
        self.source_agn_band_error = Quasar(
            source_dict=source_dict3, cosmo=None, **kwargs_quasar, **source_dict3
        )

    def test_light_curve(self):
        light_curve = self.source.light_curve
        light_curve_none = self.source_none.light_curve
        assert "i" in light_curve.keys()
        assert "r" in light_curve.keys()
        assert "MJD" in light_curve["i"].keys()
        assert "ps_mag_i" in light_curve["i"].keys()
        assert "MJD" in light_curve["r"].keys()
        assert "ps_mag_r" in light_curve["r"].keys()
        assert len(light_curve["i"]["MJD"]) == 1000

        assert light_curve_none == {}
        with pytest.raises(ValueError):
            self.source_cosmo_error.light_curve
        with pytest.raises(ValueError):
            self.source_agn_band_error.light_curve

    def test_point_source_magnitude(self):
        assert self.source.point_source_magnitude("i") == 20
        with pytest.raises(ValueError):
            self.source.point_source_magnitude("g")
        with pytest.raises(ValueError):
            self.source_none.point_source_magnitude("i", image_observation_times=10)
        assert self.source_none.point_source_magnitude("i") == 20
        npt.assert_almost_equal(
            self.source_light_curve.point_source_magnitude("i"), 20.833, decimal=2
        )


def test_extract_agn_kwargs_from_source_dict():
    source_dict = {
        "z": 0.8,
        "ps_mag_i": 20,
        "random_seed": 42,
        "black_hole_mass_exponent": 8.0,
        "eddington_ratio": 0.5,
    }
    # source_dict = Table(source_dict)

    agn_kwargs = extract_agn_kwargs_from_source_dict(source_dict=source_dict)
    assert "black_hole_mass_exponent" in agn_kwargs
    assert "eddington_ratio" in agn_kwargs
    assert agn_kwargs["black_hole_mass_exponent"] == 8.0
    assert agn_kwargs["eddington_ratio"] == 0.5


if __name__ == "__main__":
    pytest.main()
