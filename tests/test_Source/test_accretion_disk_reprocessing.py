import numpy as np
from slsim.Sources.SourceVariability.accretion_disk_reprocessing import (
    AccretionDiskReprocessing,
)
from slsim.Util.astro_util import (
    generate_signal_from_bending_power_law,
    generate_signal_from_generic_psd,
)
import pytest


class TestAccretionDiskReprocessing:
    def test_initialization(self):
        kwargs_agn_model = {
            "r_out": 1000,
            "r_resolution": 500,
            "inclination_angle": 0,
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.0,
            "eddington_ratio": 0.1,
            "corona_height": 10,
        }
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        assert reprocessor.reprocessing_model == "lamppost"

    def test_initialization_invalid_model(self):
        kwargs_agn_model = {
            "r_out": 1000,
            "r_resolution": 500,
            "inclination_angle": 0,
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.0,
            "eddington_ratio": 0.1,
            "corona_height": 10,
        }

        with pytest.raises(ValueError) as excinfo:
            AccretionDiskReprocessing("something_else", **kwargs_agn_model)
        assert (
            "Given model is not supported. Currently supported model is lamppost."
        ) in str(excinfo.value)

    def test_default_initialization_lamppost_model(self):
        kwargs_agn_model = {"r_out": 1000}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        assert reprocessor.kwargs_model["r_out"] == 1000
        assert reprocessor.kwargs_model["r_resolution"] == 500
        assert reprocessor.kwargs_model["inclination_angle"] == 0.0
        assert reprocessor.kwargs_model["black_hole_mass_exponent"] == 8.0
        assert reprocessor.kwargs_model["black_hole_spin"] == 0.0
        assert reprocessor.kwargs_model["eddington_ratio"] == 0.1

    def test_define_new_response_function(self):
        kwargs_agn_model = {"r_out": 1000}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        response_500 = reprocessor.define_new_response_function(500)
        assert len(response_500) > 0
        assert response_500.ndim == 1

    def test_define_intrinsic_signal(self):
        kwargs_agn_model = {"r_out": 1000}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        signal_output = reprocessor.define_intrinsic_signal()
        assert signal_output == (None, None)

        kwargs_signal = {
            "length_of_light_curve": 100,
            "time_resolution": 1,
            "log_breakpoint_frequency": -1,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "mean_magnitude": 3,
            "standard_deviation": 1,
            "seed": 55,
        }

        time_array, magnitude_array = generate_signal_from_bending_power_law(
            **kwargs_signal
        )

        reprocessor.define_intrinsic_signal(time_array, magnitude_array)

        signal_output = reprocessor.define_intrinsic_signal()
        assert len(signal_output[0]) == len(signal_output[1])

    def test_intrinsic_signal_errors(self):
        kwargs_agn_model = {"r_out": 1000}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        time_array = [0, 10, 40]
        magnitude_array = [200, 201, 196, 154]

        with pytest.raises(ValueError) as excinfo:
            reprocessor.define_intrinsic_signal(time_array, magnitude_array)
        assert ("Input time_array and magnitude_array must be of equal length.") in str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            reprocessor.define_intrinsic_signal(magnitude_array=magnitude_array)
        assert ("You must provide both the time_array and the magnitude_array.") in str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            reprocessor.define_intrinsic_signal(time_array=time_array)
        assert ("You must provide both the time_array and the magnitude_array.") in str(
            excinfo.value
        )

    def test_reprocess_signal_errors(self):
        kwargs_agn_model = {"black_hole_mass_exponent": 9.5}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)

        input_freq = np.linspace(1 / 100, 1 / 2, 100)
        input_psd = input_freq ** (-2)

        kwargs_signal = {
            "length_of_light_curve": 100,
            "time_resolution": 1,
            "input_frequencies": input_freq,
            "input_psd": input_psd,
            "seed": 37,
        }

        with pytest.raises(ValueError) as excinfo:
            reprocessor.reprocess_signal()
        assert (
            "Please provide the intrinsic signal first, using define_intrinsic_signal()."
        ) in str(excinfo.value)

        time_array, magnitude_array = generate_signal_from_generic_psd(**kwargs_signal)
        reprocessor.define_intrinsic_signal(time_array, magnitude_array)

        with pytest.raises(ValueError) as excinfo:
            reprocessor.reprocess_signal()
        assert ("Please provide a wavelength or a response function.") in str(
            excinfo.value
        )

        with pytest.raises(ValueError) as excinfo:
            reprocessor.reprocess_signal(
                rest_frame_wavelength_in_nanometers=500,
                response_function_amplitudes=[0, 100, 100],
            )
        assert (
            "Please provide only a wavelength or only a response function. Not both!"
        ) in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            reprocessor.reprocess_signal(
                response_function_amplitudes=[1, 0.7, 0.8],
                response_function_time_lags=[0, 1, 2, 3, 4, 5],
            )
        assert (
            "The time lag array and response function array must match in length."
        ) in str(excinfo.value)

    def test_reprocessing_a_signal(self):
        kwargs_agn_model = {"black_hole_mass_exponent": 9.5}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)

        input_freq = np.linspace(1 / 100, 1 / 2, 100)
        input_psd = input_freq ** (-2)

        kwargs_signal = {
            "length_of_light_curve": 100,
            "time_resolution": 1,
            "input_frequencies": input_freq,
            "input_psd": input_psd,
            "seed": 1,
        }

        time_array, magnitude_array = generate_signal_from_generic_psd(**kwargs_signal)
        reprocessor.define_intrinsic_signal(time_array, magnitude_array)

        reprocessed_signal_500 = reprocessor.reprocess_signal(
            rest_frame_wavelength_in_nanometers=500
        )
        reprocessed_signal_1000 = reprocessor.reprocess_signal(
            rest_frame_wavelength_in_nanometers=1000
        )
        reprocessed_signal_2000 = reprocessor.reprocess_signal(
            rest_frame_wavelength_in_nanometers=2000
        )

        assert len(reprocessed_signal_500) == len(time_array)
        assert len(reprocessed_signal_1000) == len(reprocessed_signal_500)
        assert len(reprocessed_signal_1000) == len(reprocessed_signal_2000)
        # Assessing the variance of the reprocessed signal, should decrease with wavelength!
        assert reprocessed_signal_1000.var() < reprocessed_signal_500.var()
        assert reprocessed_signal_2000.var() < reprocessed_signal_1000.var()
        # Assessing a simple reprocess where the light curve is simply shifted
        reprocessed_signal_shift = reprocessor.reprocess_signal(
            response_function_time_lags=[0, 1, 2, 3, 4, 5],
            response_function_amplitudes=[0, 0, 0, 0, 0, 1],
        )
        assert all(reprocessed_signal_shift[5:] - magnitude_array[:-5] == 0)
        # Assessing 'identity' shift. Note the convolution requires at least 2 points in the kernel
        reprocessed_signal_shift = reprocessor.reprocess_signal(
            response_function_time_lags=[0, 1],
            response_function_amplitudes=[1, 0],
        )
        assert all(reprocessed_signal_shift - magnitude_array == 0)
        # Assessing that we can only feed in a list of amplitudes, allowing the method to
        # assume time units in R_g / c.
        reprocessed_signal_test = reprocessor.reprocess_signal(
            response_function_amplitudes=1 - np.linspace(-1, 1, 100) ** 2
        )
        assert len(reprocessed_signal_test) == len(magnitude_array)
