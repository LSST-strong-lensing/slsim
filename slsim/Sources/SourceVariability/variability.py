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
            agn_kwargs = {}
            driving_signal_kwargs = {}
            reprocessing_kwargs = {}
            signal_kwargs = {}
            driving_variability_model = ""
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
                    agn_kwargs[kwarg] = self.kwargs_model[kwarg]
                elif kwarg in ["time_array", "magnitude_array"]:
                    signal_kwargs[kwarg] = self.kwargs_model[kwarg]
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
                    driving_signal_kwargs[kwarg] = self.kwargs_model[kwarg]
                elif kwarg in ["redshift"]:
                    self.redshift = self.kwargs_model[kwarg]
                elif kwarg in ["delta_wavelength"]:
                    self.delta_wavelength = self.kwargs_model[kwarg]
                elif kwarg in ["driving_variability_model"]:
                    driving_variability_model = self.kwargs_model[kwarg]
                else:
                    reprocessing_kwargs[kwarg] = self.kwargs_model[kwarg]

            accretion_disk_reprocessor = AccretionDiskReprocessing(
                "lamppost", **agn_kwargs
            )

            if "time_array" in signal_kwargs and "magnitude_array" in signal_kwargs:
                accretion_disk_reprocessor.define_intrinsic_signal(**signal_kwargs)
            else:
                driving_signal = Variability(
                    driving_variability_model, **driving_signal_kwargs
                )
                if "time_array" not in signal_kwargs:

                    signal_kwargs["time_array"] = np.linspace(
                        0,
                        driving_signal_kwargs["length_of_light_curve"] - 1,
                        driving_signal_kwargs["length_of_light_curve"],
                    )

                signal_kwargs["magnitude_array"] = driving_signal.variability_at_time(
                    signal_kwargs["time_array"]
                )
                accretion_disk_reprocessor.define_intrinsic_signal(**signal_kwargs)

            reprocessed_signals = {}
            counter = 1
            for kwarg in reprocessing_kwargs:
                if kwarg == "obs_frame_wavelength_in_nm":
                    if not isinstance(reprocessing_kwargs[kwarg], list):
                        reprocessing_kwargs[kwarg] = [reprocessing_kwargs[kwarg]]
                    for wavelength in reprocessing_kwargs[kwarg]:
                        rest_wavelength = wavelength / (1 + self.redshift)
                        reprocessed_signals[
                            "obs_wavelength_" + str(wavelength)[:6] + str("_nm")
                        ] = accretion_disk_reprocessor.reprocess_signal(
                            rest_frame_wavelength_in_nanometers=rest_wavelength
                        )

                elif kwarg == "rest_frame_wavelength_in_nm":
                    if not isinstance(reprocessing_kwargs[kwarg], list):
                        reprocessing_kwargs[kwarg] = [reprocessing_kwargs[kwarg]]
                    for rest_wavelength in reprocessing_kwargs[kwarg]:
                        reprocessed_signals[
                            "rest_wavelength_" + str(rest_wavelength)[:6] + str("_nm")
                        ] = accretion_disk_reprocessor.reprocess_signal(
                            rest_frame_wavelength_in_nanometers=rest_wavelength
                        )

                elif kwarg == "speclite_filter":
                    if not isinstance(reprocessing_kwargs[kwarg], list):
                        reprocessing_kwargs[kwarg] = [reprocessing_kwargs[kwarg]]

                    for speclite_filter in reprocessing_kwargs[kwarg]:
                        response_function = accretion_disk_reprocessor.define_passband_response_function(
                            speclite_filter,
                            redshift=self.redshift,
                            delta_wavelength=self.delta_wavelength,
                        )
                        reprocessed_signals[speclite_filter] = (
                            accretion_disk_reprocessor.reprocess_signal(
                                response_function_amplitudes=response_function
                            )
                        )

                elif kwarg == "response_function":
                    if not isinstance(
                        reprocessing_kwargs[kwarg][0], list
                    ) and not isinstance(reprocessing_kwargs[kwarg][0], np.ndarray):
                        reprocessing_kwargs[kwarg] = [reprocessing_kwargs[kwarg]]

                    if not isinstance(
                        reprocessing_kwargs[kwarg][0][0], list
                    ) and not isinstance(reprocessing_kwargs[kwarg][0][0], np.ndarray):
                        reprocessing_kwargs[kwarg] = [reprocessing_kwargs[kwarg]]

                    for response in reprocessing_kwargs[kwarg]:
                        if len(response) == 1:
                            reprocessed_signals[kwarg + "_" + str(counter)] = (
                                accretion_disk_reprocessor.reprocess_signal(
                                    response_function_amplitudes=response[0]
                                )
                            )
                            counter += 1
                        elif len(response) == 2:
                            reprocessed_signals[kwarg + "_" + str(counter)] = (
                                accretion_disk_reprocessor.reprocess_signal(
                                    response_function_time_lags=response[0],
                                    response_function_amplitudes=response[1],
                                )
                            )
                            counter += 1
                        else:
                            raise ValueError(
                                "response function must be defined by "
                                "one or two lists / arrays. Not more!"
                            )

            self._model = []
            for key, signal in reprocessed_signals.items():
                light_curve = {
                    "MJD": accretion_disk_reprocessor.time_array,
                    "ps_mag_" + key: signal,
                }
                light_curve_class = LightCurveInterpolation(light_curve)
                self._model.append([key, light_curve_class.magnitude])

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
        if isinstance(self._model, list):
            signal_dict = {}
            for light_curve in self._model:
                label = light_curve[0]
                signal_dict[label] = light_curve[1](observation_times)
            return signal_dict
        return self._model(observation_times)
