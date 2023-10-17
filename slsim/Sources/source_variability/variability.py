import numpy as np
import astropy.units as u

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
        self._variability_model = variability_model
        if self._variability_model not in ["sinusoidal"]:
            raise ValueError(
                "given model is not supported. Currently,"
                "supported model is sinusoudal."
            )
        if self._variability_model == "sinusoidal":
            self._model = sinusoidal_variability
        else:
            raise ValueError("Please provide a supported variability model.")

        self._kwargs_model = kwargs_variability_model

    def variability_at_t(self, observation_times):
        """Provides variability of a source at given time.

        :param observation_times: image observation time
        :return: variability at given time.
        """
        return self._model(observation_times, **self._kwargs_model)


def sinusoidal_variability(x, **kwargs_model):
    """Calculate the sinusoidal variability for a given observation time.

    :param x: observation time (astropy.unit object, e.g., 3*u.day, 3*u.second).
    :param kwargs_model: dictionary of variability parameter associated with a source.
    :return: variability for the given time
    """
    unit = x.unit
    if unit == u.day:
        t = x
    else:
        t = x.to(u.day)
    amplitude = kwargs_model.get("amp", 1.0)
    frequency = kwargs_model.get("freq", 1.0)

    return amplitude * np.sin(2 * np.pi * frequency * t.value)
