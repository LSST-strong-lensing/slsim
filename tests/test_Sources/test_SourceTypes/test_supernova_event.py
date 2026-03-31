from slsim.Sources.SourceTypes.supernova_event import SupernovaEvent
import numpy as np
import pytest
from astropy import cosmology
import slsim.ImageSimulation.image_quality_lenstronomy as iql


class TestSupernovaEvent:
    def setup_method(self):
        self.cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}
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

        self.source = SupernovaEvent(cosmo=self.cosmo, **kwargs_sn, **self.source_dict)
        self.source_roman = SupernovaEvent(
            cosmo=self.cosmo, **kwargs_sn_roman, **self.source_dict
        )
        self.source_none = SupernovaEvent(
            cosmo=self.cosmo, **kwargs_sn_none, **source_dict2
        )
        self.source_cosmo_error = SupernovaEvent(
            cosmo=None, **kwargs_sn, **self.source_dict
        )
        self.source_light_curve = SupernovaEvent(
            cosmo=self.cosmo, **kwargs_sn_none, **source_dict3
        )

    def test_light_curve(self):
        light_curve = self.source.light_curve
        light_curve_roman = self.source_roman.light_curve
        light_curve_none = self.source_none.light_curve

        # Check that the non-band parameters are successfully ignored
        assert "supernovae_lightcurve" not in light_curve.keys()
        assert "supernovae_lightcurve" not in light_curve_roman.keys()

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

    def test_light_curve_warning(self):
        """Test that a UserWarning is raised when a band is supported by SLSim
        but missing in sncosmo."""

        # register a dummy observatory with a fake band so SLSim recognizes it but sncosmo does not
        class DummyObs:
            def __init__(self, band, **kwargs):
                pass

            def kwargs_single_band(self):
                return {}

        iql.register_observatory("DummyObs", DummyObs, bands=["unregistered_sn_band"])

        # modify the source to request this fake band
        self.source._kwargs_variability = [
            "supernovae_lightcurve",
            "unregistered_sn_band",
        ]

        # sncosmo doesn't know about "unregistered_sn_band", so it should raise the warning and skip it
        with pytest.warns(UserWarning, match="Failed to generate lightcurve"):
            failed_light_curve = self.source.light_curve

        assert failed_light_curve == {}

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
