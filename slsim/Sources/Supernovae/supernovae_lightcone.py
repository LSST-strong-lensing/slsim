from skypy.galaxies.redshift import redshifts_from_comoving_density
from slsim.Sources.Supernovae.supernovae_pop import SNIaRate
from astropy import units


class SNeLightcone(object):
    """Class to integrate SNe comoving density n(z) in light cone volume."""

    def __init__(self, cosmo, redshifts, sky_area, noise, time_interval):
        """
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        :param redshifts: redshifts for supernovae density lightcone to be evaluated at
        :type redshifts: array-like
        :param sky_area: sky area for sampled galaxies in [solid angle]
        :type sky_area: `~Astropy.units.Quantity`
        :param noise: poisson-sample the number of galaxies in supernovae density lightcone
        :type noise: bool
        :param time_interval: time interval for supernovae density lightcone to be evaluated over
        :type time_interval: `~Astropy.units.Quantity`
        """
        self._cosmo = cosmo
        self._input_redshifts = redshifts
        self._sky_area = sky_area
        self._noise = noise
        self._time_interval = time_interval

        sne_rate = SNIaRate(self._cosmo, self._input_redshifts[-1])
        h = self._cosmo.H(0).to_value() / 100

        # Account for included factor of h.
        self.density = self.convert_density(
            sne_rate.calculate_SNIa_rate(self._input_redshifts) * h
        )

    def convert_density(self, density):
        """Converts SN Ia comoving densities from [yr^(-1)Mpc^(-3)] to have the
        desired time unit.

        :param density: initial comoving density of SN Ia
            [yr^(-1)Mpc^(-3)]
        :return: SN Ia comoving density with the desired time unit
            [day^(-1)Mpc^(-3), hr^(-1)Mpc^(-3), etc.]
        """
        time_conversion = (1 * units.year).to(self._time_interval.unit)
        converted_density = density / time_conversion * self._time_interval.value
        return converted_density.value

    def supernovae_sample(self):
        """Integrates SNe comoving density in light cone.

        :return: sampled redshifts such that the comoving number density
            of galaxies corresponds to the input distribution
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
