import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

"""References:
Oguri and Marshall 2010
"""


def calculate_star_formation_rate(z):
    """Calculates the cosmic star formation rate. (Eq 13 - Oguri and Marshall 2010)

    :param z: redshift
    :return: cosmic star formation rate in [(h)yr^(-1)Mpc^(-3)]
    """
    if z >= 0:
        star_formation_rate = (0.0118 + 0.08 * z) / (1 + (z / 3.3) ** 5.2)
        return star_formation_rate
    else:
        return np.nan


def t_d_power(t_d):
    """Calculates the power law constraint on time delay. (Eq 14 - Oguri and Marshall 2010)

    :param t_d: time delay
    :return: constrained time delay
    """
    if t_d > 0:
        ft_d = (t_d) ** (-1.08)
        return ft_d
    else:
        return np.nan


class SNIaRate(object):
    """Class to calculate supernovae rates."""

    def __init__(self, cosmo):
        """
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        """
        self._cosmo = cosmo

    def z_from_time(self, t):
        """Calculates redshift given cosmic time.

        :param t: cosmic time since big bang in [Gy]
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        :return: redshift at time t [float]
        """
        if not hasattr(self, "_age_inv"):
            z_array = np.linspace(0, 1000, 100000)
            z_array = z_array[::-1]
            t_array = self._cosmo.age(z_array).to_value()
            t_array = np.insert(t_array, 0, -20)
            z_array = np.insert(z_array, 0, 1000000)
            self._age_inv = interp.interp1d(t_array, z_array)

        return self._age_inv(t).item()

    def _numerator_integrand(self, t_d, t):
        """Calculates the numerator integrand to be used within calculate_SNIa_rates.
        (Eq 15 - Oguri and Marshall 2010)

        :param t_d: time delay
        :param t: lookback time
        :return: numerator integrand
        """
        if t < 0 or t_d < 0:
            return np.nan
        else:
            ft_d = t_d_power(t_d)
            z_t = self.z_from_time(t - t_d)
            return calculate_star_formation_rate(z_t) * ft_d

    def calculate_SNIa_rate(self, z, eta=0.04):
        """Calculates the rate of SN Ia. (Eq 15 - Oguri and Marshall 2010)

        :param z: redshift
        :param eta: canonical efficiency
        :param cosmo: cosmology object
        :type cosmo: ~astropy.cosmology object
        :return: SN Ia rate n(z) in [(h)yr^(-1)Mpc^(-3)]
        """
        if z < 0:
            return np.nan
        else:
            C_Ia = 0.032

            t_z = self._cosmo.age(z).to_value()  # Time at given redshift z
            t_0 = self._cosmo.age(0).to_value()  # Time at redshift z = 0
            t = self._cosmo.lookback_time(
                z
            ).to_value()  # Lookback time at given redshift z

            numerator = integrate.quad(
                lambda t_d: self._numerator_integrand(t_d, t), 0.1, t_z
            )
            denominator = integrate.quad(lambda t_d: t_d_power(t_d), 0.1, t_0)

            SNIa_rate = eta * C_Ia * (numerator[0] / denominator[0])
            return SNIa_rate
