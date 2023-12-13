from slsim.Sources.SourceVariability.light_curve_interpolation import \
    LightCurveInterpolation
from slsim.Sources.SourceVariability.sinusoidal_variability import SinusoidalVariability

"""This class aims to have realistic variability models for AGN and supernovae."""

class Variability(object):
    def __init__(self, variability_model, **kwargs_variability_model):
        """Initialize the variability class.

        :param variability_model: keyword for variability model to be used.
        :type variability_model: str
        :param kwargs_variability_model: Keyword arguments for variability class. For
            sinusoidal_variability kwargs are amplitude ('amp') and frequency ('freq').
        :type kwargs_variability_model: dict
        """
        self.kwargs_model = kwargs_variability_model
        self.variability_model = variability_model
        if self.variability_model == "sinusoidal":
            sinusoidal_class = SinusoidalVariability(**self.kwargs_model)
            self._model = sinusoidal_class.magnitude
        elif self.variability_model == "light_curve":
            #Here, we extract light curve from kwargs_model and feed them to 
            # LightCurveInterpolation class to get interpolation function of magnitude.
            time_array = self.kwargs_model["MJD"]
            string = "ps_mag_"
            magnitude_values = {key: value for key, 
                        value in self.kwargs_model.items() if key.startswith(string)}
            magnitude_array = list(magnitude_values.values())[0]
            light_curve_class = LightCurveInterpolation(times = time_array, 
                                                 magnitudes = magnitude_array)
            self._model = light_curve_class.magnitude
        else:
            raise ValueError(
                "Given model is not supported. Currently supported models are" 
                "sinusoidal, light_curve."
            )

    def variability_at_time(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        return self._model(observation_times)
