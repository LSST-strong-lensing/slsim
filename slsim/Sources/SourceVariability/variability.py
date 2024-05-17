from slsim.Sources.SourceVariability.light_curve_interpolation import (
    LightCurveInterpolation,
)
from slsim.Sources.SourceVariability.sinusoidal_variability import SinusoidalVariability
from slsim.Sources.SourceVariability.accretion_disk_reprocessing import (
    AccretionDiskReprocessing,
)
from slsim.Util.astro_util import generate_signal_from_bending_power_law
from slsim.Util.astro_util import generate_signal_from_generic_psd
import numpy as np

"""This class aims to have realistic variability models for AGN and supernovae."""


class Variability(object):
    def __init__(self, variability_model, **kwargs_variability_model):
        """Initialize the variability class.

        :param variability_model: keyword for variability model to be used.
        :type variability_model: str
        :param kwargs_variability_model: Keyword arguments for variability class.
            For sinusoidal_variability kwargs are: amplitude ('amp') and frequency ('freq').
            For bending_power_law kwargs are: ('length_of_light_curve'), ('time_resolution'),
                ('log_breakpoint_frequency'), ('low_frequency_slope'),
                ('high_frequency_slope'), ('mean_amplitude'),
                ('standard_deviation'), ('normal_magnitude_variance'), ('zero_point_mag'), and ('seed')
            For user_defined_psd kwargs are: ('length_of_light_curve'), ('time_resolution'),
                ('input_frequencies'), ('input_psd'), ('mean_amplitude'),
                ('standard_deviation'), ('normal_magnitude_variance'), ('zero_point_mag'), and ('seed')
            For lamppost_reprocessed kwargs are:
                - all kwargs for AccretionDiskReprocessing model
                - one of the following two options:
                    1) all kwargs for another Variability object to use as the
                        driving signal with additional keyword ('driving_variability_model')
                    2) ('time_array') and ('magnitude_array') to define the driving signal
                - ('redshift') to bring observer frame wavelengths to the rest frame
                - ('delta_wavelength') to change the resolution (in nm) of a speclite filter
                - ('obs_frame_wavelength_in_nm') list of observer frame wavelengths in nm
                - ('rest_frame_wavelength_in_nm') list of rest frame wavelengths in nm
                - ('speclite_filter') list of speclite filter names to use
                - ('response_function') list of response functions to use (see notebook for samples)

        :type kwargs_variability_model: dict
        """
        self.kwargs_model = kwargs_variability_model
        self.variability_model = variability_model
        if self.variability_model == "sinusoidal":
            sinusoidal_class = SinusoidalVariability(**self.kwargs_model)
            self._model = sinusoidal_class.magnitude

        elif self.variability_model == "light_curve":
            light_curve_class = LightCurveInterpolation(light_curve=self.kwargs_model)
            self._model = light_curve_class.magnitude

        elif self.variability_model == "bending_power_law":
            time_array, magnitude_array = generate_signal_from_bending_power_law(
                **self.kwargs_model
            )
            light_curve = {"MJD": time_array, "ps_mag_arbitrary": magnitude_array}
            light_curve_class = LightCurveInterpolation(light_curve)
            self._model = light_curve_class.magnitude

        elif self.variability_model == "user_defined_psd":
            time_array, magnitude_array = generate_signal_from_generic_psd(
                **self.kwargs_model
            )
            light_curve = {"MJD": time_array, "ps_mag_arbitrary": magnitude_array}
            light_curve_class = LightCurveInterpolation(light_curve)
            self._model = light_curve_class.magnitude

        elif self.variability_model == "lamppost_reprocessed":
            self.parse_kwargs_for_lamppost_reprocessed_model()

            self.accretion_disk_reprocessor = AccretionDiskReprocessing(
                "lamppost", **self.agn_kwargs
            )

            if (
                "time_array" in self.signal_kwargs
                and "magnitude_array" in self.signal_kwargs
            ):
                self.accretion_disk_reprocessor.define_intrinsic_signal(
                    **self.signal_kwargs
                )
            else:
                driving_signal = Variability(
                    self.kwargs_model["driving_variability_model"],
                    **self.driving_signal_kwargs
                )
                if "time_array" not in self.signal_kwargs:
                    self.signal_kwargs["time_array"] = np.linspace(
                        0,
                        self.driving_signal_kwargs["length_of_light_curve"] - 1,
                        self.driving_signal_kwargs["length_of_light_curve"],
                    )

                self.signal_kwargs["magnitude_array"] = (
                    driving_signal.variability_at_time(self.signal_kwargs["time_array"])
                )
                self.accretion_disk_reprocessor.define_intrinsic_signal(
                    **self.signal_kwargs
                )

            light_curve = self.reprocess_with_lamppost_model()
            light_curve_class = LightCurveInterpolation(light_curve=light_curve)
            self._model = light_curve_class.magnitude

        else:
            raise ValueError(
                "Given model is not supported. Currently supported models are "
                "sinusoidal, light_curve, bending_power_law, "
                "user_defined_psd, lamppost_reprocessed."
            )

    def variability_at_time(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        return self._model(observation_times)

    def parse_kwargs_for_lamppost_reprocessed_model(self):
        """Separates categories of self.kwargs for lamppost reprocessing.

        :return: Dictionaries for:
            - agn kwargs
            - driving signal kwargs
            - reprocessing kwargs
            - signal kwargs
        """
        self.agn_kwargs = {}
        self.driving_signal_kwargs = {}
        self.reprocessing_kwargs = {}
        self.signal_kwargs = {}

        self.redshift = 0
        self.delta_wavelength = 50

        for kwarg in self.kwargs_model:
            if kwarg in [
                "r_out",
                "corona_height",
                "r_resolution",
                "inclination_angle",
                "black_hole_mass_exponent",
                "black_hole_spin",
                "eddington_ratio",
            ]:
                self.agn_kwargs[kwarg] = self.kwargs_model[kwarg]

            elif kwarg in ["time_array", "magnitude_array"]:
                self.signal_kwargs[kwarg] = self.kwargs_model[kwarg]

            elif kwarg in [
                "length_of_light_curve",
                "time_resolution",
                "log_breakpoint_frequency",
                "low_frequency_slope",
                "high_frequency_slope",
                "mean_magnitude",
                "standard_deviation",
                "normal_magnitude_variance",
                "zero_point_mag",
                "input_frequencies",
                "input_psd",
                "light_curve",
                "seed",
                "amp",
                "freq",
            ]:
                self.driving_signal_kwargs[kwarg] = self.kwargs_model[kwarg]

            elif kwarg in ["redshift"]:
                self.redshift = self.kwargs_model[kwarg]
            elif kwarg in ["delta_wavelength"]:
                self.delta_wavelength = self.kwargs_model[kwarg]
            else:
                self.reprocessing_kwargs[kwarg] = self.kwargs_model[kwarg]

    def reprocess_with_lamppost_model(self):
        """Reprocesses the signal based on the type of accretion disk reprocessor.

        :return: dict containing a light curve object parameters
        """

        if "obs_frame_wavelength_in_nm" in self.reprocessing_kwargs:
            wavelength = self.reprocessing_kwargs["obs_frame_wavelength_in_nm"]
            rest_wavelength = wavelength / (1 + self.redshift)

            reprocessed_signal = self.accretion_disk_reprocessor.reprocess_signal(
                rest_frame_wavelength_in_nanometers=rest_wavelength
            )
            light_curve = {
                "MJD": self.signal_kwargs["time_array"],
                "ps_mag_" + str(wavelength)[:5]: reprocessed_signal,
            }

        elif "rest_frame_wavelength_in_nm" in self.reprocessing_kwargs:
            rest_wavelength = self.reprocessing_kwargs["rest_frame_wavelength_in_nm"]
            reprocessed_signal = self.accretion_disk_reprocessor.reprocess_signal(
                rest_frame_wavelength_in_nanometers=rest_wavelength
            )
            light_curve = {
                "MJD": self.signal_kwargs["time_array"],
                "ps_mag_" + str(rest_wavelength)[:5]: reprocessed_signal,
            }

        elif "speclite_filter" in self.reprocessing_kwargs:
            speclite_filter = self.reprocessing_kwargs["speclite_filter"]
            response_function = (
                self.accretion_disk_reprocessor.define_passband_response_function(
                    speclite_filter,
                    redshift=self.redshift,
                    delta_wavelength=self.delta_wavelength,
                )
            )
            reprocessed_signal = self.accretion_disk_reprocessor.reprocess_signal(
                response_function_amplitudes=response_function
            )
            light_curve = {
                "MJD": self.signal_kwargs["time_array"],
                "ps_mag_" + str(speclite_filter): reprocessed_signal,
            }
        elif "response_function_amplitudes" in self.reprocessing_kwargs:
            if "response_function_time_lags" not in self.reprocessing_kwargs:
                reprocessed_signal = self.accretion_disk_reprocessor.reprocess_signal(
                    response_function_amplitudes=self.reprocessing_kwargs[
                        "response_function_amplitudes"
                    ]
                )
            else:
                reprocessed_signal = self.accretion_disk_reprocessor.reprocess_signal(
                    response_function_amplitudes=self.reprocessing_kwargs[
                        "response_function_amplitudes"
                    ],
                    response_function_time_lags=self.reprocessing_kwargs[
                        "response_function_time_lags"
                    ],
                )
            light_curve = {
                "MJD": self.signal_kwargs["time_array"],
                "ps_mag_user": reprocessed_signal,
            }
        else:
            raise ValueError("Please provide a reprocessing method")

        return light_curve
