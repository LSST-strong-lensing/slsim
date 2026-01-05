from slsim.Sources.SourceTypes.quasar import Quasar, extract_agn_kwargs_from_source_dict
import numpy as np
import numpy.testing as npt
import pytest
from astropy import cosmology
from slsim.Pipelines import roman_speclite


class TestQuasar:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = {"z": 0.8, "ps_mag_i": 20}
        source_dict2 = {
            "z": 0.8,
            "MJD": [0, 2, 3, 4, 5, 6],
            "ps_mag_i": [21, 20, 18, 21, 22, 23],
        }
        source_dict3 = {"z": 0.8}

        # Source dict specifically for testing AGN parameter extraction/updates
        source_dict_agn = {
            "z": 0.8,
            "ps_mag_i": 20,
            "black_hole_mass_exponent": 8.5,
            "eddington_ratio": 0.1,
            "random_seed": 42,
        }

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
            "kwargs_variability": {"agn_lightcurve", "i", "r", "F062", "VIS"},
            "agn_driving_variability_model": "bending_power_law",
            "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
            "lightcurve_time": np.linspace(0, 1000, 1000),
            "corona_height": 10,
            "r_resolution": 500,
        }

        kwargs_quasar_none = {
            "source_type": "quasar",
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "agn_driving_variability_model": "bending_power_law",
            "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
            "lightcurve_time": np.linspace(0, 1000, 1000),
            "corona_height": 10,
            "r_resolution": 500,
        }

        # run roman_speclite to load the Roman filters
        roman_speclite.configure_roman_filters()

        self.source = Quasar(cosmo=cosmo, **source_dict, **kwargs_quasar)

        self.source_none = Quasar(cosmo=cosmo, **kwargs_quasar_none, **source_dict)
        self.source_cosmo_error = Quasar(cosmo=None, **kwargs_quasar, **source_dict)
        self.source_light_curve = Quasar(
            cosmo=cosmo, **kwargs_quasar_none, **source_dict2
        )
        self.source_agn_band_error = Quasar(
            source_dict=source_dict3, cosmo=None, **kwargs_quasar, **source_dict3
        )
        # Initialize the source with explicit AGN parameters for morphology testing
        self.source_agn_params = Quasar(cosmo=cosmo, **source_dict_agn, **kwargs_quasar)

    def test_light_curve(self):
        light_curve = self.source.light_curve
        light_curve_none = self.source_none.light_curve

        # Test LSST bands
        assert "i" in light_curve.keys()
        assert "r" in light_curve.keys()
        assert "MJD" in light_curve["i"].keys()
        assert "ps_mag_i" in light_curve["i"].keys()
        assert "MJD" in light_curve["r"].keys()
        assert "ps_mag_r" in light_curve["r"].keys()
        assert len(light_curve["i"]["MJD"]) == 1000

        # Test Roman bands
        assert "F062" in light_curve.keys()
        assert "ps_mag_F062" in light_curve["F062"].keys()

        # Test Euclid bands
        assert "VIS" in light_curve.keys()
        assert "ps_mag_VIS" in light_curve["VIS"].keys()

        assert light_curve_none == {}
        with pytest.raises(ValueError):
            self.source_cosmo_error.light_curve
        with pytest.raises(ValueError):
            self.source_agn_band_error.light_curve

    def test_point_source_magnitude(self):
        # Test basic LSST magnitude
        assert self.source.point_source_magnitude("i") == 20

        # Test that calling point_source_magnitude triggers variability computation
        # and allows access to Roman/Euclid mean magnitudes if time is None
        # Values will differ from i-band due to SED, just checking it returns a float
        roman_mag = self.source.point_source_magnitude("F062")
        assert isinstance(roman_mag, float)
        assert roman_mag != 20  # Should be different from i-band magnitude

        with pytest.raises(ValueError):
            self.source.point_source_magnitude("g")
        with pytest.raises(ValueError):
            self.source_none.point_source_magnitude("i", image_observation_times=10)
        assert self.source_none.point_source_magnitude("i") == 20
        npt.assert_almost_equal(
            self.source_light_curve.point_source_magnitude("i"), 20.833, decimal=2
        )

    def test_update_microlensing_kwargs_source_morphology(self):
        # We must trigger the light curve generation to ensure the internal agn_class is initialized
        _ = self.source_agn_params.light_curve

        initial_kwargs = {"some_other_param": 123}
        updated_kwargs = (
            self.source_agn_params.update_microlensing_kwargs_source_morphology(
                initial_kwargs
            )
        )

        # Check that original parameters are preserved
        assert updated_kwargs["some_other_param"] == 123

        # Check that AGN parameters provided in setup_method are correctly added
        assert "black_hole_mass_exponent" in updated_kwargs
        assert updated_kwargs["black_hole_mass_exponent"] == 8.5
        assert "eddington_ratio" in updated_kwargs
        assert updated_kwargs["eddington_ratio"] == 0.1


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
