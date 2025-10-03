import numpy as np
from slsim.Sources.SourceVariability.accretion_disk_reprocessing import (
    AccretionDiskReprocessing,
)
from slsim.Util.astro_util import (
    generate_signal_from_bending_power_law,
    generate_signal_from_generic_psd,
)
import pytest
import astropy.units as u


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
        assert reprocessor.redshift == 0

        kwargs_agn_model["redshift"] = 3.1
        redshifted_reprocessor = AccretionDiskReprocessing(
            "lamppost", **kwargs_agn_model
        )
        assert redshifted_reprocessor.redshift == 3.1

        with pytest.raises(ValueError):
            AccretionDiskReprocessing("other", **kwargs_agn_model)

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

        kwargs_agn_model["redshift"] = 3.1
        AccretionDiskReprocessing("lamppost", **kwargs_agn_model)

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
        kwargs_small_agn_model = {"black_hole_mass_exponent": 4.0}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        small_reprocessor = AccretionDiskReprocessing(
            "lamppost", **kwargs_small_agn_model
        )

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
        small_reprocessor.define_intrinsic_signal(time_array, magnitude_array)

        reprocessed_signal_500 = reprocessor.reprocess_signal(
            rest_frame_wavelength_in_nanometers=500
        )
        reprocessed_signal_1000 = reprocessor.reprocess_signal(
            rest_frame_wavelength_in_nanometers=1000
        )
        reprocessed_signal_2000 = reprocessor.reprocess_signal(
            rest_frame_wavelength_in_nanometers=2000
        )
        small_reprocessor.reprocess_signal(rest_frame_wavelength_in_nanometers=500)

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
        small_reprocessor.reprocess_signal(
            response_function_amplitudes=1 - np.linspace(-1, 1, 100) ** 2
        )

        time_array_list = [1, 2, 3, 4, 5, 8, 20]
        magnitude_array_list = [8, 5, 4, 3, 2, 1, 0]
        small_reprocessor.define_intrinsic_signal(
            time_array=time_array_list,
            magnitude_array=magnitude_array_list,
        )
        small_reprocessor.reprocess_signal(
            response_function_time_lags=[0, 1],
            response_function_amplitudes=[1, 0],
        )

    def test_define_passband_response_function(self):
        kwargs_agn_model = {"black_hole_mass_exponent": 9.5}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)
        lsst_filter = "lsst2016-r"
        filter_response = reprocessor.define_passband_response_function(
            lsst_filter,
            redshift=0,
            delta_wavelength=50,
            passband_wavelength_unit=u.angstrom,
        )
        wavelength_in_nm = 500
        wavelength_response = reprocessor.define_new_response_function(
            wavelength_in_nm,
        )
        assert len(wavelength_response) == len(filter_response)

        reprocessor.define_passband_response_function(
            lsst_filter,
            redshift=0,
            delta_wavelength=5,
            passband_wavelength_unit=u.angstrom,
        )

    def test_determine_agn_luminosity_from_known_luminosity(self):
        kwargs_agn_model = {"black_hole_mass_exponent": 9.5}
        reprocessor = AccretionDiskReprocessing("lamppost", **kwargs_agn_model)

        i_band_magnitude = 20
        known_band = "lsst2023-i"
        redshift = 1
        mag_zero_point = 0
        unknown_band = "lsst2023-r"
        wavelength = 700

        r_band_magnitude = reprocessor.determine_agn_luminosity_from_known_luminosity(
            known_band, i_band_magnitude, redshift, mag_zero_point, band=unknown_band
        )

        reprocessor.determine_agn_luminosity_from_known_luminosity(
            known_band,
            i_band_magnitude,
            redshift,
            mag_zero_point,
            observer_frame_wavelength_in_nm=wavelength,
        )

        # test identiy
        i_band_mag = reprocessor.determine_agn_luminosity_from_known_luminosity(
            known_band, i_band_magnitude, redshift, mag_zero_point, band="lsst2023-i"
        )
        assert i_band_mag == 20

        # test recipricosity
        test_i_band_magnitude = (
            reprocessor.determine_agn_luminosity_from_known_luminosity(
                unknown_band,
                r_band_magnitude,
                redshift,
                mag_zero_point,
                band=known_band,
            )
        )
        # Use numpy testing because of rounding
        np.testing.assert_almost_equal(i_band_magnitude, test_i_band_magnitude)

        # test errors
        with pytest.raises(ValueError):
            reprocessor.determine_agn_luminosity_from_known_luminosity(
                known_band, i_band_magnitude, redshift, mag_zero_point
            )
        with pytest.raises(ValueError):
            reprocessor.determine_agn_luminosity_from_known_luminosity(
                100,
                i_band_magnitude,
                redshift,
                mag_zero_point,
                band=known_band,
            )
