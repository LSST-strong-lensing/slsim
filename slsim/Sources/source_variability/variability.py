import numpy as np
import scipy

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

def interpolate_variability(Movie, Orig_timestamps, New_timestamps):
    """Interpolates a variable source to any given time series

    :param Movie: 3 dimensional array of shape (t, x, y)
        relating to snapshots of a variable object in source plane
    :param Orig_timestamps: Series of timestamps which represent original snapshots
    :param New_timestamps: Series of new timestamps to interpolate to
    :return: Linearly interpolated movie along time axis
    """
    initial_shape = np.shape(Movie)
    npix = initial_shape[1] * initial_shape[2]
    space_positions = np.linspace(1, npix, npix)                        # Order pixels
    intermediate_movie = np.reshape(Movie, (initial_shape[0], npix))    # Reshape into (t, space) array
    interpolation = scipy.interpolate.RegularGridInterpolator((Orig_timestamps, space_positions),
                                        intermediate_movie, bounds_error=False, fill_value=None)
    new_points = np.meshgrid(New_timestamps, space_positions, indexing='ij')
    movie_resampled = interpolation((new_points[0], new_points[1]))     # Resample
    return np.reshape(movie_resampled, (np.size(New_timestamps), initial_shape[1], initial_shape[2]))
