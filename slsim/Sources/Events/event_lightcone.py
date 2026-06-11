from skypy.galaxies.redshift import redshifts_from_comoving_density
from slsim.Sources.Events.event_pop import EventPopulation
from astropy import units
import numpy as np


class EventLightcone(object):
    def __init__(self, cosmo, redshifts, sky_area, noise, time_interval, model):
        """
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        :param redshifts: redshifts for BNS merger density lightcone to be evaluated at
        :type redshifts: array-like
        :param sky_area: sky area for sampled event in [solid angle]
        :type sky_area: `~Astropy.units.Quantity`
        :param noise: poisson-sample the number of event in supernovae density lightcone
        :type noise: bool
        :param time_interval: time interval for supernovae density lightcone to be evaluated over
        :type time_interval: `~Astropy.units.Quantity`
        :param model : name of model, chosen form "BNS" or "SNIa"
        :type model: string value
        """
        self._cosmo = cosmo
        self._input_redshifts = np.asarray(redshifts)
        self._sky_area = sky_area
        self._noise = noise
        self._time_interval = time_interval
        self._model = model

        event_pop = EventPopulation(self._model, self._cosmo, self._input_redshifts[-1])

        # Convert source-frame event rate to observer-frame event rate
        rate_source_frame = event_pop.event_rate(self._input_redshifts)
        rate_observer_frame = rate_source_frame / (1 + self._input_redshifts)

        self.density = self.convert_density(rate_observer_frame)

    def convert_density(self, density):
        """Converts event comoving densities from [yr^(-1)Mpc^(-3)] to have the
        desired time unit.

        :param density: initial comoving density of event, such as BNS
            merger or SNIa [yr^(-1)Mpc^(-3)]
        :return: BNS merger comoving density with the desired time unit
            [day^(-1)Mpc^(-3), hr^(-1)Mpc^(-3), etc.]
        """
        time_conversion = (1 * units.year).to(self._time_interval.unit)
        converted_density = density / time_conversion * self._time_interval.value

        return converted_density.value

    def event_sample(self):
        """Integrates event comoving density in light cone.

        :return: sampled redshifts such that the comoving number density
            of events corresponds to the input distribution
        :return type: numpy.ndarray
        """
        if not hasattr(self, "_output_redshifts"):
            self._output_redshifts = redshifts_from_comoving_density(
                redshift=self._input_redshifts,
                density=self.density,
                sky_area=self._sky_area,
                cosmology=self._cosmo,
                noise=self._noise,
            )
        return self._output_redshifts
