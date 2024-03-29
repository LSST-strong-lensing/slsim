from scipy.interpolate import interp1d


class LightCurveInterpolation(object):
    """This class manages interpolation of light curve of a source."""

    def __init__(self, light_curve):
        """
        :param light_cureve: dictionary containg observation time and magnitude of a
         point source. Eg: light_curve = {"MJD": np.array([20, 30, 40, 50, 60, 70, 80]),
         "ps_mag_i": np.array([25, 24, 23, 20, 21, 23, 30])}
        """
        self.light_curve = light_curve
        string = "ps_mag_"
        time_array = self.light_curve["MJD"]
        magnitude_values = {
            key: value
            for key, value in self.light_curve.items()
            if key.startswith(string)
        }
        magnitude_array = list(magnitude_values.values())[0]
        self.interpolation = interp1d(
            time_array,
            magnitude_array,
            kind="linear",
            fill_value=(magnitude_array[0], magnitude_array[-1]),
            bounds_error=False,
        )

    def magnitude(self, observation_time):
        """Provides magnitude at given time.

        :param observation_time: observation time of a source in days
        :type observation_time: float
        :return: magnitude at given observation time
        """
        return self.interpolation(observation_time)
