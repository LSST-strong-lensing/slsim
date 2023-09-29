import numpy as np
import astropy.units as u

"""This class aims to have realistic variability models for AGN and Supernovae."""

class Variability:
    def __init__(self, **kwargs_variability):
        """
        Initialize the variability class.

        :param kwargs: Keyword arguments for variability class. For 
         sinusoidal_variability kwargs are amplitude ('amp') and frequency ('freq').
        """
        self.kwargs_variability = kwargs_variability

    def sinusoidal_variability(self, x):
        """
        Calculate the sinusoidal variability for a given observation time.

        :param x: observation time (astropy.unit object, e.g., 3*u.day, 3*u.second).
        :return: variability for the given time
        """
        unit = x.unit
        if unit == u.day:
            t=x
        else:
            t = x.to(u.day)
        amplitude = self.kwargs_variability.get('amp', 1.0)
        frequency = self.kwargs_variability.get('freq', 1.0)
        
        return amplitude * np.sin(2 * np.pi * frequency * t.value)
