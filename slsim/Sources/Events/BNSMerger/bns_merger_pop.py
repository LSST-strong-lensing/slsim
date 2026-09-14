import numpy as np
import scipy.integrate as integrate
from slsim.Util.cosmo_util import z_time_interp

"""References:
  SNIa population: Oguri and Marshall 2010
  BNS merger population: Kuwahara et al. 2025
"""


def norm_delay_time_distribution(t_d, t_d_min, t_d_max):
    """Calculates the normalized time delay. (Described in text after Eq 3 - Kuwahara et al. 2025)

    :param t_d: time delay (t_d>=0) in [Gyr]
    :param t_d_min: minimum of t_d in [Gyr]
    :param t_d_max: maximum of t_d in [Gyr]

    :return: probability distribution of the time delay
    """

    ft_d = 1 / (t_d * (np.log(t_d_max / t_d_min)))
    return ft_d


class BNSMergerRate(object):
    """Class to calculate BNS merger rates."""

    def __init__(self, cosmo, z_max):
        """
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        :param z_max: maximum redshift to describe the SFR density (no stars accounted for >z_max)
        """
        self._cosmo = cosmo
        self._z_max = z_max

        self._t_min = self._cosmo.age(self._z_max).to_value()  # Time at redshift z_max
        self._t_0 = self._cosmo.age(0).to_value()  # Time at redshift z = 0

        self._t_d_min = 0.020  # minimum of t_d in [Gyr], assumed as 20 Myr
        self._t_d_max = 13.8  # maximum of t_d in [Gyr], assumed as Hubble time

        self._z_from_time = z_time_interp(self._cosmo, self._z_max)

        self._local_merger_rate = 320 * 1e-9  # in [yr^(-1)Mpc^(-3)]

    def binary_formation_rate(self, z):
        """Calculates the binary formation rate. (Eq 3 - Kuwahara et al. 2025)

        ``nu``, ``z_m``, ``a``, and ``b`` are fixed fitting parameters for
        ``R_f(z)`` in Eq 3 from Kuwahara et al. 2025.

        :param z: redshift (z>=0)

        :return: binary formation rate in [(M_sol)yr^(-1)Gpc^(-3)]
        """
        a = 2.80
        b = 2.46
        z_m = 1.72
        nu = 0.146  # in unit of M_sol * Gpc^(-3) * yr^(-1)

        binary_formation_rate = (nu * a * np.exp(b * (z - z_m))) / (
            a - b + b * np.exp(a * (z - z_m))
        )
        return binary_formation_rate

    def _numerator_integrand(self, t_d, t):
        """Calculates the numerator integrand to be used within calculate_event_rates.
        (Eq 4 - Kuwahara et al. 2025)

        :param t_d: time delay in [Gyr]. It must satisfy ``t_d < t``.
        :param t: Cosmic time at the merger redshift in [Gyr].

        :return: numerator integrand
        """
        ft_d = norm_delay_time_distribution(
            t_d, t_d_min=self._t_d_min, t_d_max=self._t_d_max
        )
        z_f = self._z_from_time(t - t_d).item()  # formation redshift
        return self.binary_formation_rate(z_f) * ft_d

    def event_rate(self, z):
        """Calculates the normalized BNS merger rate R_m(z) in source frame, which is
        calibrated by the local merger rate Rm(z = 0). (Eq 4 - Kuwahara et al. 2025)

        :param z: an array of redshift (z>=0). No need to be sorted.

        :return: BNS merger rate R_m(z) in [yr^(-1)Mpc^(-3)]
        :return type: array-like
        """

        unorm_BNS_rate_list = []

        for i in z:
            t_z = self._cosmo.age(i).to_value()  # Time at given redshift z
            upper_limit = t_z - self._t_min

            if upper_limit <= self._t_d_min:
                unorm_BNS_rate_list.append(0.0)
                continue

            numerator = integrate.quad(
                self._numerator_integrand,
                self._t_d_min,
                t_z - self._t_min,
                args=(t_z,),
                limit=1000,
                epsabs=1e-8,
                epsrel=1e-6,
            )
            unorm_BNS_rate_list.append(numerator[0])

        unorm_BNS_rate_array = np.array(unorm_BNS_rate_list)

        # Calculate unnormalized BNS merger rate at z=0 for normalization
        unorm_BNS_rate_z0 = integrate.quad(
            self._numerator_integrand,
            self._t_d_min,
            self._t_0 - self._t_min,
            args=(self._t_0,),
        )[0]
        BNS_rate_list = (
            unorm_BNS_rate_array / unorm_BNS_rate_z0
        ) * self._local_merger_rate

        return np.array(BNS_rate_list)
