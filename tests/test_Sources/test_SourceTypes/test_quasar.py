from slsim.Sources.SourceTypes.quasar import Quasar, extract_agn_kwargs_from_source_dict
import numpy as np
import numpy.testing as npt
import pytest
from astropy import cosmology
from slsim.Pipelines import roman_speclite


class TestQuasar:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = {"z": 0.8, "ps_mag_i": 20}

        # Dictionary simulating a light curve input
        source_dict2 = {
            "z": 0.8,
            "MJD": [0, 2, 3, 4, 5, 6],
            "ps_mag_i": [21, 20, 18, 21, 22, 23],
        }
        self.source_dict3 = {"z": 0.8}

        # Source dict specifically for testing AGN parameter extraction/updates
        source_dict_agn = {
            "z": 0.8,
            "ps_mag_i": 20,
            "black_hole_mass_exponent": 8.5,
            "eddington_ratio": 0.1,
            "random_seed": 42,
        }

        # Source dict simulating input from QSOGen (pre-computed multi-band mags)
        # We explicitly set 'g' to a value (15) that is likely very different
        # from what the AGN model would predict based on i=20.
        source_dict_multiband = {"z": 0.8, "ps_mag_i": 20, "ps_mag_g": 15.0}

        variable_agn_kwarg_dict = {
            "length_of_light_curve": 500,
            "time_resolution": 1,
            "log_breakpoint_frequency": 1 / 20,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "standard_deviation": 0.9,
        }
        self.kwargs_quasar = {
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

        # Configuration for multiband test: request 'g' band variability
        kwargs_quasar_multiband = self.kwargs_quasar.copy()
        kwargs_quasar_multiband["kwargs_variability"] = {"agn_lightcurve", "g", "i"}

        # run roman_speclite to load the Roman filters
        roman_speclite.configure_roman_filters()

        self.source = Quasar(cosmo=cosmo, **self.source_dict, **self.kwargs_quasar)
        self.source_none = Quasar(cosmo=cosmo, **kwargs_quasar_none, **self.source_dict)
        # Objects with missing cosmology or bands should raise errors when methods are called (lazy initialization)
        self.source_cosmo_error = Quasar(
            cosmo=None, **self.kwargs_quasar, **self.source_dict
        )
        self.source_agn_band_error = Quasar(
            source_dict=self.source_dict3,
            cosmo=cosmo,
            **self.kwargs_quasar,
            **self.source_dict3
        )

        self.source_light_curve = Quasar(
            cosmo=cosmo, **kwargs_quasar_none, **source_dict2
        )
        # Initialize the source with explicit AGN parameters for morphology testing
        self.source_agn_params = Quasar(
            cosmo=cosmo, **source_dict_agn, **self.kwargs_quasar
        )

        # Source with pre-existing magnitudes
        self.source_multiband = Quasar(
            cosmo=cosmo, **source_dict_multiband, **kwargs_quasar_multiband
        )

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

        # Test that runtime access raises ValueError for invalid cosmology
        with pytest.raises(ValueError):
            _ = self.source_cosmo_error.light_curve

        # Test that runtime access raises ValueError for missing bands/magnitudes
        with pytest.raises(ValueError):
            _ = self.source_agn_band_error.light_curve

    def test_light_curve_with_existing_mags(self):
        """Test that if source_dict already contains magnitudes (e.g. from
        qsogen), Quasar class uses them as the mean magnitude for variability,
        OVERRIDING the SS73 model prediction."""
        light_curve = self.source_multiband.light_curve

        # Verify Variability was computed for 'g'
        assert "g" in light_curve

        # Check the mean magnitude used in the reprocessing
        # The light curve should fluctuate around the INPUT magnitude (15.0),
        # not the SS73 model magnitude (which would be ~20 like the i-band).
        mean_g_mag = np.mean(light_curve["g"]["ps_mag_g"])

        # Allow small deviation due to stochastic variability, but it should be close to 15
        npt.assert_allclose(mean_g_mag, 15.0, atol=0.5)

        # Verify it is NOT close to 20 (standard SS73 model prediction)
        assert abs(mean_g_mag - 20.0) > 1.0

    def test_point_source_magnitude(self):
        # Test basic LSST magnitude (static check)
        # This calls super().point_source_magnitude which checks source_dict
        assert self.source.point_source_magnitude("i") == 20

        # Test that calling point_source_magnitude WITH time triggers variability
        # and populates the source_dict with mean magnitudes for other bands.

        # 1. Before computation, "F062" is unknown
        with pytest.raises(ValueError):
            # Fails because F062 is not in source_dict and we didn't provide time
            # to trigger calculation
            self.source.point_source_magnitude("F062")

        # 2. Provide time -> triggers computation
        times = np.array([10, 20])
        roman_mags = self.source.point_source_magnitude(
            "F062", image_observation_times=times
        )

        assert len(roman_mags) == 2

        # 3. After computation, mean magnitude is added to source_dict,
        # so static call should now work
        roman_static = self.source.point_source_magnitude("F062")
        assert isinstance(roman_static, float)
        assert roman_static != 20  # Should be different from i-band magnitude

        # Test error for unsupported band
        with pytest.raises(ValueError):
            # 'u' band was not in kwargs_variability for self.source setup
            self.source.point_source_magnitude("u", image_observation_times=times)

        # Test pass-through for non-variable source
        assert self.source_none.point_source_magnitude("i") == 20

        # Test interpolation logic from source_light_curve (pre-defined lightcurve)
        npt.assert_almost_equal(
            self.source_light_curve.point_source_magnitude("i"), 20.833, decimal=2
        )

    def test_update_microlensing_kwargs_source_morphology(self):
        # agn_class is now initialized lazily inside update_microlensing...

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

    agn_kwargs = extract_agn_kwargs_from_source_dict(source_dict=source_dict)
    assert "black_hole_mass_exponent" in agn_kwargs
    assert "eddington_ratio" in agn_kwargs
    assert agn_kwargs["black_hole_mass_exponent"] == 8.0
    assert agn_kwargs["eddington_ratio"] == 0.5


if __name__ == "__main__":
    pytest.main()
