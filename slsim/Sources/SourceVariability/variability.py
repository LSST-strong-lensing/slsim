import numpy as np
from scipy.interpolate import interp1d

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
        self.variability_model = variability_model
        if self.variability_model not in ["sinusoidal", "light_curve"]:
            raise ValueError(
                "Given model is not supported. Currently supported model is sinusoidal."
            )
        self.kwargs_model = kwargs_variability_model

    def variability_at_time(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        if self.variability_model == "sinusoidal":
            self._model = sinusoidal_variability
            variability_result = self._model(observation_times, **self.kwargs_model)
        elif self.variability_model == "light_curve":
            self._model = light_curve_interpolation
            interp_function = self._model(**self.kwargs_model)
            variability_result = interp_function(observation_times)
        else:
            variability_result = None
        return variability_result

def sinusoidal_variability(t, amp, freq):
    """Calculate the sinusoidal variability for a given observation time.

    :param t: observation time in [day].
    :param kwargs_model: dictionary of variability parameter associated with a source.
    :return: variability for the given time
    """
    return amp * abs(np.sin(2 * np.pi * freq * t))

def light_curve_interpolation(**kwargs_model):
    """Interpolates provided light curves. For this function, it is not necessary to 
    specify particular band while using in variability class so that one has freedom to 
    work with any band.
    
    :param kwargs_model: kwargs_model is a light curve in any band. It should contain 
     magnitude in any band and corresponding observational time in days.
    :return: interpolated function for magnitude  
    """ 
    band_string = "ps_mag_"
    time = kwargs_model["MJD"]
    magnitude_values = {key: value for key, 
                        value in kwargs_model.items() if key.startswith(band_string)}
    magnitude = list(magnitude_values.values())[0]
    interp_function = interp1d(time, magnitude, kind='linear', fill_value='extrapolate')
    return interp_function
