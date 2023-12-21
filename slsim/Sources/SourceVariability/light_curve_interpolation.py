from scipy.interpolate import interp1d

class LightCurveInterpolation(object):
    """This class manages interpolation of light curve of a source"""
    def __init__(self, times, magnitudes):
        """
        :param times: array of observation time
        :param magnitudes: array of magnitude
        """
        self.time_array = times
        self.magnitude_array = magnitudes
        self.interpolation = interp1d(self.time_array, self.magnitude_array, 
                                kind='linear', fill_value='extrapolate')

    def magnitude(self, observation_time):
        """Provides magnitude at given a time.

        :param observation_time: observation time of a source
        :return: magnitude at given observation time  
        """ 
        return self.interpolation(observation_time)