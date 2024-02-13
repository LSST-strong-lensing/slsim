import numpy as np
from scipy.stats import skewnorm
"""Class to generate fake light curve"""

class FakeLightCurve:
    def __init__(self, cosmo):
        """
        :param cosmo: astropy cosmology object
        """
        self.cosmo = cosmo
    def generate_light_curve(self, redshift, peak_magnitude,  num_points=50, time_range=(0, 100)):
        """Generates a fake light curve

        :param redshift: redshift of an object
        :param peak_magnitude: peak magnitude of the light curve
        :param num_points: number of data points in light curve
        :param time_range: range of time
        :return: lightcurve
        """
        peak_time=np.random.randint(20, 45)
        sigma=30
        skewness = 15
        time = np.linspace(time_range[0], time_range[1], num_points)
        apparent_magnitudes = 10 + 5 * np.log10(self.cosmo.luminosity_distance(redshift).value / 10) + \
                              peak_magnitude * skewnorm.pdf(time, skewness, loc=peak_time, scale=sigma)
        return time, apparent_magnitudes