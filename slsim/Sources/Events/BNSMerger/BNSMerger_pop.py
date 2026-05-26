import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

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

        self._t_min = self._cosmo.age(
            z=self._z_max
        ).to_value()  # Time at redshift z_max
        self._t_0 = self._cosmo.age(0).to_value()  # Time at redshift z = 0

        self.t_d_min = 0.020  # minimum of t_d in [Gyr], assumed as 20 Myr
        self.t_d_max = 13.8  # maximum of t_d in [Gyr], assumed as Hubble time

    def calculate_binary_formation_rate(self, z):
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

    def z_from_time(self, t):
        """Calculates redshift given cosmic time.

        :param t: cosmic time since big bang in [Gyr]
        :return: redshift at time t [float]
        """
        if not hasattr(self, "_age_inv"):
            z_array = np.linspace(0, self._z_max, 100)
            z_array = z_array[::-1]
            t_array = self._cosmo.age(z_array).to_value()
            self._age_inv = interp.interp1d(t_array, z_array, fill_value="extrapolate")
        return self._age_inv(t).item()

    def _numerator_integrand(self, t_d, t):
        """Calculates the numerator integrand to be used within calculate_event_rates.
        (Eq 4 - Kuwahara et al. 2025)

        :param t_d: time delay in [Gyr]. It must satisfy ``t_d < t``.
        :param t: Cosmic time at the merger redshift in [Gyr].

        :return: numerator integrand
        """
        ft_d = norm_delay_time_distribution(t_d, t_d_min=self.t_d_min, t_d_max=self.t_d_max)
        z_t = self.z_from_time(t - t_d)  # time since big bang
        return self.calculate_binary_formation_rate(z_t) * ft_d

    def calculate_event_rate(self, z):
        """Calculates the rate of events, such as BNS merger. (Eq 4 - Kuwahara et al. 2025)

        :param z: an array of redshift (z>=0). No need to be sorted.

        :return: BNS merger rate R_m(z) in [(M_sol)yr^(-1)Gpc^(-3)], following the unit of R_f
        :return type: array-like
        """
        BNS_rate_list = []

        for i in z:
            t_z = self._cosmo.age(i).to_value()  # Time at given redshift z

            numerator = integrate.quad(
                self._numerator_integrand,
                self.t_d_min,
                t_z - self._t_min,
                args=(t_z,),
                limit=200,
                epsabs=1e-6,
                epsrel=1e-4,
            )
            BNS_rate_list.append(numerator[0])

        return np.array(BNS_rate_list)
