import numpy as np
import sncosmo
from astropy import units as u


class SimpleSupernovaLightCurve:
    def __init__(self, cosmo):
        """
        :param cosmo: astropy cosmology object
        """
        self.cosmo = cosmo

    def generate_light_curve(
        self,
        redshift,
        absolute_magnitude,
        num_points=50,
        lightcurve_time=50 * u.day,
        band="r",
    ):
        """Generates a simple light curve.

        :param redshift: redshift of an object
        :param absolute_magnitude: absolute magnitude of the source
        :param num_points: number of data points in light curve
        :param time_range: range of time
        :return: lightcurve
        """
        time_range = (-20, lightcurve_time.to(u.day).value)
        time = np.linspace(time_range[0], time_range[1], num_points)

        model = sncosmo.Model(source="salt2")
        model.set(z=redshift)
        model.set_source_peakabsmag(absolute_magnitude, "bessellr", "ab")

        if band == "r":
            band_name = "bessellr"
        elif band == "i":
            band_name = "besselli"
        elif band == "g":
            band_name = "bessellg"
        else:
            raise ValueError("Provided band is not supported")

        flux = model.bandflux(band_name, time)
        apparent_magnitudes = -2.5 * np.log10(flux)

        return time, apparent_magnitudes
