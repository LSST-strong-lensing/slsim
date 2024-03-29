import numpy as np


class SinusoidalVariability(object):
    """This class manages sinusoidal variability of a source."""

    def __init__(self, amp, freq):
        """
        :param amp: amplitude for a given source
        :param freq: frequency of a given source
        """
        self.amp = amp
        self.freq = freq

    def magnitude(self, observation_time):
        """Provides magnitude at a given observation time.

        :param observation_time: observation time in [day].
        :return: magnitude for the given time
        """
        return self.amp * abs(np.sin(2 * np.pi * self.freq * observation_time))
