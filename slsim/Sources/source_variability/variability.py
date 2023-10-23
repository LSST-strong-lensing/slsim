import numpy as np

"""This class aims to have realistic variability models for AGN and Supernovae."""


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
        if self.variability_model not in ["sinusoidal"]:
            raise ValueError(
                "Given model is not supported. Currently supported model is sinusoidal."
            )
        if self.variability_model == "sinusoidal":
            self._model = sinusoidal_variability
        else:
            raise ValueError("Please provide a supported variability model.")

        self.kwargs_model = kwargs_variability_model

    def variability_at_time(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        return self._model(observation_times, **self.kwargs_model)


def sinusoidal_variability(t, amp, freq):
    """Calculate the sinusoidal variability for a given observation time.

    :param t: observation time in [day].
    :param kwargs_model: dictionary of variability parameter associated with a source.
    :return: variability for the given time
    """
    return amp * np.sin(2 * np.pi * freq * t)
