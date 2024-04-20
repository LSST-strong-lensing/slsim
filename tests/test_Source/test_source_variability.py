import numpy as np
from slsim.Sources.SourceVariability.variability import Variability
from numpy import testing as npt
import pytest


class TestVariability:
    def test_initialization_valid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        assert variability.variability_model == "sinusoidal"

    def test_light_curve_variability(self):
        mjd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ps_mag_1 = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        observation_times = np.array([1, 2, 3, 4, 5])
        var = Variability(variability_model="light_curve", MJD=mjd, ps_mag_1=ps_mag_1)
        results = var.variability_at_time(observation_times)
        assert np.all(results) == np.all(ps_mag_1)

    def test_initialization_invalid_model(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        with pytest.raises(ValueError) as excinfo:
            Variability("invalid_model", **kwargs_model)
        assert (
            "Given model is not supported. Currently supported models are "
            "sinusoidal, light_curve, bending_power_law, "
            "user_defined_psd, lamppost_reprocessed."
        ) in str(excinfo.value)

    def test_variability_at_t_sinusoidal(self):
        kwargs_model = {"amp": 1.0, "freq": 0.5}
        variability = Variability("sinusoidal", **kwargs_model)
        observation_times = np.array([np.pi, np.pi / 2, np.pi / 3])
        result = variability.variability_at_time(observation_times)
        expected_result = np.array([0.43030122, 0.97536797, 0.14773276])
        npt.assert_almost_equal(result, expected_result, decimal=5)

    def test_variability_bending_power_law(self):
        kwargs_model = {
            "length_of_light_curve": 1000,
            "time_resolution": 1,
            "log_breakpoint_frequency": -2,
            "mean_magnitude": 0,
            "seed": None,
        }
        var = Variability("bending_power_law", **kwargs_model)
        observation_times = np.linspace(0, 1000, 1000)
        results = var.variability_at_time(observation_times)
        assert var.variability_model == "bending_power_law"
        npt.assert_almost_equal(results.mean(), 0)

    def test_variability_user_defined_psd(self):
        frequencies = np.linspace(1 / 1000, 1 / 2, 1000)
        psd = frequencies ** (-3)
        kwargs_model = {
            "length_of_light_curve": 1000,
            "time_resolution": 1,
            "input_frequencies": frequencies,
            "input_psd": psd,
            "mean_magnitude": 50,
            "seed": None,
        }
        var = Variability("user_defined_psd", **kwargs_model)
        observation_times = np.linspace(0, 999, 1000)
        results = var.variability_at_time(observation_times)
        assert var.variability_model == "user_defined_psd"
        npt.assert_almost_equal(results.mean(), 50)

    def test_lamppost_reprocessed(self):
        agn_kwargs = {
            "r_out": 1000,
            "corona_height": 10,
            "r_resolution": 100,
            "inclination_angle": 40,
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.0,
            "eddington_ratio": 0.1,
        }
        signal_kwargs = {
            "time_array": [0, 1, 2, 3, 4, 5, 6, 7],
            "magnitude_array": [5, 3, 1, 3, 6, 9, 12, 10],
        }
        bpl_kwargs = {
            "length_of_light_curve": 100,
            "time_resolution": 1,
            "log_breakpoint_frequency": -2,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "mean_magnitude": 10,
            "standard_deviation": 0.2,
            "normal_magnitude_variance": True,
            "zero_point_mag": 200,
            "seed": 17,
            "driving_variability_model": "bending_power_law",
        }
        freq = np.linspace(1, 1000, 1000)
        psd = freq ** (-2)
        user_def_signal_kwargs = {
            "length_of_light_curve": 100,
            "time_resolution": 1,
            "input_frequencies": freq,
            "input_psd": psd,
            "mean_magnitude": 10,
            "standard_deviation": 0.2,
            "normal_magnitude_variance": True,
            "zero_point_mag": 200,
            "seed": 17,
            "driving_variability_model": "user_defined_psd",
        }
        reprocessing_kwargs = {
            "obs_frame_wavelength_in_nm": 50,
            "rest_frame_wavelength_in_nm": 300,
            "speclite_filter": "lsst2016-r",
        }
        multi_reprocessing_kwargs = {
            "obs_frame_wavelength_in_nm": [50, 100],
            "rest_frame_wavelength_in_nm": [300, 20],
            "speclite_filter": ["lsst2016-r", "lsst2016-g"],
        }
        other_kwargs = {
            "redshift": 1.0,
            "delta_wavelength": 100,
        }
        with pytest.raises(ValueError):
            Variability("lamppost_reprocessed", **{})
            Variability("lamppost_reprocessed", **agn_kwargs)
            Variability("lamppost_reprocessed", **signal_kwargs)
            Variability("lamppost_reprocessed", **reprocessing_kwargs)
        # Test three different types of fully defined responses
        full_kwargs_1 = {}
        for dictionary in [
            agn_kwargs,
            signal_kwargs,
            reprocessing_kwargs,
            other_kwargs,
        ]:
            for key in dictionary:
                full_kwargs_1[key] = dictionary[key]
        full_kwargs_2 = {}
        for dictionary in [
            agn_kwargs,
            bpl_kwargs,
            multi_reprocessing_kwargs,
            other_kwargs,
        ]:
            for key in dictionary:
                full_kwargs_2[key] = dictionary[key]
        full_kwargs_3 = {}
        for dictionary in [
            agn_kwargs,
            user_def_signal_kwargs,
            multi_reprocessing_kwargs,
            other_kwargs,
        ]:
            for key in dictionary:
                full_kwargs_3[key] = dictionary[key]
        Variability("lamppost_reprocessed", **full_kwargs_1)
        Variability("lamppost_reprocessed", **full_kwargs_2)
        Variability("lamppost_reprocessed", **full_kwargs_3)
        # Test minimum case of a fully defined response
        time_array = np.linspace(0, 10, 10)
        magnitude_array = (10 - time_array) * time_array
        min_kwargs = {"time_array": time_array, "magnitude_array": magnitude_array}
        Variability("lamppost_reprocessed", **min_kwargs)


if __name__ == "__main__":
    pytest.main()
