from slsim.Sources.SourceVariability.light_curve_interpolation import (
    LightCurveInterpolation,
)
from slsim.Sources.SourceVariability.sinusoidal_variability import SinusoidalVariability
from slsim.Util.astro_util import generate_signal_from_bending_power_law
from slsim.Util.astro_util import generate_signal_from_generic_psd

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

        else:
            raise ValueError(
                "Given model is not supported. Currently supported models are"
                "sinusoidal, light_curve, bending_power_law, user_defined_psd."
            )

    def variability_at_time(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        return self._model(observation_times)
