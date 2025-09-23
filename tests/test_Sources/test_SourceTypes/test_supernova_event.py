from slsim.Sources.SourceTypes.supernova_event import SupernovaEvent
import numpy as np
import pytest
from astropy import cosmology


class TestSupernovaEvent:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}
        source_dict2 = {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005, "ps_mag_i": 20}
        source_dict3 = {
            "z": 0.8,
            "ra_off": 0.001,
            "dec_off": 0.005,
            "MJD": [0, 2, 3, 4, 5, 6],
            "ps_mag_i": [21, 20, 19, 21, 22, 23],
        }
        kwargs_sn = {
            "source_type": "supernova",
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i", "r"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None,
        }
        kwargs_sn_roman = {
            "source_type": "supernova",
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "F062", "F129"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 100),
            "sn_modeldir": None,
        }

        kwargs_sn_none = {
            "source_type": "supernova",
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 100),
            "sn_modeldir": None,
        }

        self.source = SupernovaEvent(cosmo=cosmo, **kwargs_sn, **source_dict)
        self.source_roman = SupernovaEvent(
            cosmo=cosmo, **kwargs_sn_roman, **source_dict
        )
        self.source_none = SupernovaEvent(cosmo=cosmo, **kwargs_sn_none, **source_dict2)
        self.source_cosmo_error = SupernovaEvent(cosmo=None, **kwargs_sn, **source_dict)
        self.source_light_curve = SupernovaEvent(
            cosmo=cosmo, **kwargs_sn_none, **source_dict3
        )

    def test_light_curve(self):
        light_curve = self.source.light_curve
        light_curve_roman = self.source_roman.light_curve
        light_curve_none = self.source_none.light_curve
        assert "i" in light_curve.keys()
        assert "r" in light_curve.keys()
        assert "MJD" in light_curve["i"].keys()
        assert "ps_mag_i" in light_curve["i"].keys()
        assert "MJD" in light_curve["r"].keys()
        assert "ps_mag_r" in light_curve["r"].keys()
        assert len(light_curve["i"]["MJD"]) == 150

        assert "F062" in light_curve_roman.keys()
        assert "F129" in light_curve_roman.keys()
        assert "MJD" in light_curve_roman["F062"].keys()
        assert "ps_mag_F062" in light_curve_roman["F062"].keys()
        assert "MJD" in light_curve_roman["F129"].keys()
        assert "ps_mag_F129" in light_curve_roman["F129"].keys()
        assert len(light_curve_roman["F062"]["MJD"]) == 100

        assert not light_curve_none
        with pytest.raises(ValueError):
            self.source_cosmo_error.light_curve

    def test_point_source_magnitude(self):
        # supernova is randomly generated. So, can't assert a fix number for magnitude.
        # Just checking these numbers are generated.
        assert self.source.point_source_magnitude("i") is not None
        with pytest.raises(ValueError):
            self.source.point_source_magnitude("g")
        with pytest.raises(ValueError):
            self.source_none.point_source_magnitude("i", image_observation_times=10)
        assert self.source_none.point_source_magnitude("i") == 20
        assert self.source_light_curve.point_source_magnitude("i") == 21


if __name__ == "__main__":
    pytest.main()
