from slsim.Sources.Events.BNSMerger.kilonova import Kilonova
from slsim.Sources.SourceTypes.kilonova_event import KilonovaEvent
import slsim.ImageSimulation.image_quality_lenstronomy as iql
import numpy as np
import numpy.testing as npt
import pytest
from astropy import cosmology


class TestKilonovaEvent(object):
    def setup_method(self):
        self.cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}

        source_dict2 = {
            "z": 0.8,
            "ra_off": 0.001,
            "dec_off": 0.005,
            "ps_mag_i": 20,
        }

        source_dict3 = {
            "z": 0.8,
            "ra_off": 0.001,
            "dec_off": 0.005,
            "MJD": [0, 2, 3, 4, 5, 6],
            "ps_mag_i": [21, 20, 19, 21, 22, 23],
        }

        self.kwargs_kilonova = {
            "mej_1": 0.01,
            "mej_2": 0.02,
            "mej_3": 0.03,
            "vej_1": 0.1,
            "vej_2": 0.2,
            "vej_3": 0.3,
            "kappa_1": 0.5,
            "kappa_2": 3.0,
            "kappa_3": 10.0,
            "temperature_floor_1": 5000,
            "temperature_floor_2": 4000,
            "temperature_floor_3": 3000,
            "kappa_gamma": 10,
        }

        kwargs_bns = {
            # model type
            "model_type": "bns_merger",
            "variability_model": "light_curve",
            "kwargs_variability": ["bns_lightcurve", "i", "r"],
            "lightcurve_time": np.linspace(0.1, 10, 50),
            "model_name": "mosfit_kilonova",
            "mag_zpsys": "AB",
            "modeldir": None,
            "kwargs_kilonova": self.kwargs_kilonova,
        }

        kwargs_bns_none = {
            "model_type": "bns_merger",
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "lightcurve_time": np.linspace(0.1, 10, 50),
            "model_name": "mosfit_kilonova",
            "mag_zpsys": "AB",
            "modeldir": None,
            "kwargs_kilonova": None,
        }

        self.source = KilonovaEvent(cosmo=self.cosmo, **kwargs_bns, **self.source_dict)
        self.source_none = KilonovaEvent(
            cosmo=self.cosmo, **kwargs_bns_none, **source_dict2
        )
        self.source_cosmo_error = KilonovaEvent(
            cosmo=None, **kwargs_bns, **self.source_dict
        )
        self.source_light_curve = KilonovaEvent(
            cosmo=self.cosmo, **kwargs_bns_none, **source_dict3
        )

    def test_light_curve(self):
        light_curve = self.source.light_curve
        light_curve_none = self.source_none.light_curve

        # Check that the non-band parameter is successfully ignored.
        assert "bns_lightcurve" not in light_curve.keys()

        assert "i" in light_curve.keys()
        assert "r" in light_curve.keys()
        assert "MJD" in light_curve["i"].keys()
        assert "ps_mag_i" in light_curve["i"].keys()
        assert "MJD" in light_curve["r"].keys()
        assert "ps_mag_r" in light_curve["r"].keys()
        assert len(light_curve["i"]["MJD"]) == 52
        assert len(light_curve["i"]["ps_mag_i"]) == 52

        assert not light_curve_none

        with pytest.raises(ValueError):
            self.source_cosmo_error.light_curve

        # _lightcurve_class must be stored as a Kilonova instance.
        assert isinstance(self.source._lightcurve_class, Kilonova)

        # Check that kwargs_kilonova is passed into the Kilonova class.
        assert self.source._lightcurve_class._model_parameters["mej_1"] == 0.01
        assert self.source._lightcurve_class._model_parameters["mej_2"] == 0.02
        assert self.source._lightcurve_class._model_parameters["mej_3"] == 0.03
        assert self.source._lightcurve_class._model_parameters["vej_1"] == 0.1
        assert self.source._lightcurve_class._model_parameters["vej_2"] == 0.2
        assert self.source._lightcurve_class._model_parameters["vej_3"] == 0.3
        assert self.source._lightcurve_class._model_parameters["kappa_1"] == 0.5
        assert self.source._lightcurve_class._model_parameters["kappa_2"] == 3.0
        assert self.source._lightcurve_class._model_parameters["kappa_3"] == 10.0
        assert (
            self.source._lightcurve_class._model_parameters["temperature_floor_1"]
            == 5000
        )
        assert (
            self.source._lightcurve_class._model_parameters["temperature_floor_2"]
            == 4000
        )
        assert (
            self.source._lightcurve_class._model_parameters["temperature_floor_3"]
            == 3000
        )
        assert self.source._lightcurve_class._model_parameters["kappa_gamma"] == 10

        # Check that the flux is zero before and after the modeled light curve.
        assert np.isinf(light_curve["i"]["ps_mag_i"][0])
        assert np.isinf(light_curve["i"]["ps_mag_i"][-1])

        # test specific values of a model lightcurve
        npt.assert_almost_equal(
            light_curve["i"]["ps_mag_i"][1], 30.52057182997839, decimal=5
        )
        npt.assert_almost_equal(
            np.min(light_curve["i"]["ps_mag_i"]), 28.330211764355443, decimal=5
        )
        npt.assert_almost_equal(
            light_curve["i"]["ps_mag_i"][-2], 31.496458821405547, decimal=5
        )

    def test_light_curve_warning(self):
        """Test that a UserWarning is raised when lightcurve generation
        fails."""

        class DummyObs:
            def __init__(self, band, **kwargs):
                pass

            def kwargs_single_band(self):
                return {}

        iql.register_observatory(
            "DummyBNSObs", DummyObs, bands=["unregistered_bns_band"]
        )

        self.source._kwargs_variability = [
            "bns_lightcurve",
            "unregistered_bns_band",
        ]

        with pytest.warns(UserWarning, match="Failed to generate lightcurve"):
            failed_light_curve = self.source.light_curve

        assert failed_light_curve == {}

    def test_point_source_magnitude(self):
        assert self.source.point_source_magnitude("i") is not None

        with pytest.raises(ValueError):
            self.source.point_source_magnitude("g")

        with pytest.raises(ValueError):
            self.source_none.point_source_magnitude("i", image_observation_times=10)

        assert self.source_none.point_source_magnitude("i") == 20
        assert self.source_light_curve.point_source_magnitude("i") == 21


if __name__ == "__main__":
    pytest.main()
