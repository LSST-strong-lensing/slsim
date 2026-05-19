import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

"""References:
  SNIa population: Oguri and Marshall 2010
  BNS merger population: Kuwahara et al. 2025
"""


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

        self.t_d_min = 0.020  # Gyr, 20 Myr
        self.t_d_max = 13.8  # Gyr, Hubble time approximately

    def calculate_star_formation_rate(self, z):
        """Calculates the binary formation rate. (Eq 3 - Kuwahara et al. 2025)

        :param z: redshift (z>=0)
        :param z_m: peak-related redshift parameter
        :param nu: normalization constant
        :param a & b: observable parameters by fitting the data based on the gamma-ray burst.

        :return: binary formation rate in [(M_sol)yr^(-1)Gpc^(-3)]
        """
        a = 2.80
        b = 2.46
        z_m = 1.72
        nu = 0.146  # in unit of M_sol * Gpc^(-3) * yr^(-1)

        star_formation_rate = (nu * a * np.exp(b * (z - z_m))) / (
            a - b + b * np.exp(a * (z - z_m))
        )
        return star_formation_rate

    def delay_time_distribution(self, t_d):
        """Calculates the normalized time delay. (Described in text after Eq 3 - Kuwahara et al. 2025)

        :param t_d: time delay (t_d>=0) in [Gyr]
        :param t_d_min: minimum of t_d in [Gyr], assumed as 20 Myr
        :param t_d_max: maximum of t_d in [Gyr], assumed as Hubble time

        :return: probability distribution of the time delay
        """

        ft_d = 1 / (t_d * (np.log(self.t_d_max / self.t_d_min)))
        return ft_d

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

        :param t_d: time delay (t>=0)
        :param t: time at final integrated redshift (t>=0)

        :return: numerator integrand
        """
        ft_d = self.delay_time_distribution(t_d)
        z_t = self.z_from_time(t - t_d)  # time since big bang
        return self.calculate_star_formation_rate(z_t) * ft_d

    def calculate_event_rate(self, z):
        """Calculates the rate of events, such as BNS merger. (Eq 4 - Kuwahara et al. 2025)

        :param z: redshift (z>=0)

        :return: BNS merger rate R_m(z), following the unit of R_f
        :return type: array-like
        """
        BNS_rate_list = []

        for i in z:
            t_z = self._cosmo.age(i).to_value()  # Time at given redshift z

            numerator = integrate.quad(
                self._numerator_integrand, self.t_d_min, t_z - self._t_min, args=(t_z,)
            )
            BNS_rate_list.append(numerator[0])

        return np.array(BNS_rate_list)


class SNIaRate(object):
    """Class to calculate supernovae rates."""

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

        self._denominator = integrate.quad(
            self.delay_time_distribution, 0.1, self._t_0 - self._t_min
        )

    def calculate_star_formation_rate(self, z):
        """Calculates the cosmic star formation rate. (Eq 13 - Oguri and Marshall 2010)

        :param z: redshift (z>=0)

        :return: cosmic star formation rate in [(h)(M_sol)yr^(-1)Mpc^(-3)]
        """
        star_formation_rate = (0.0118 + 0.08 * z) / (1 + (z / 3.3) ** 5.2)
        return star_formation_rate

    def delay_time_distribution(self, t_d):
        """Calculates the power-law constraint on time delay. (Eq 14 - Oguri and Marshall 2010)

        :param t_d: time delay (t_d>=0) in [Gyr]

        :return: constrained time delay
        """
        ft_d = (t_d) ** (-1.08)
        return ft_d

    def z_from_time(self, t):
        """Calculates redshift given cosmic time.

        :param t: cosmic time since big bang in [Gy]
        :return: redshift at time t [float]
        """
        if not hasattr(self, "_age_inv"):
            z_array = np.linspace(0, self._z_max, 100)
            z_array = z_array[::-1]
            t_array = self._cosmo.age(z_array).to_value()
            self._age_inv = interp.interp1d(t_array, z_array, fill_value="extrapolate")
        return self._age_inv(t).item()

    def _numerator_integrand(self, t_d, t):
        """Calculates the numerator integrand to be used within calculate_SNIa_rates.
        (Eq 15 - Oguri and Marshall 2010)

        :param t_d: time delay (t>=0)
        :param t: time at final integrated redshift (t>=0)

        :return: numerator integrand
        """
        ft_d = self.delay_time_distribution(t_d)
        z_t = self.z_from_time(t - t_d)  # time since big bang
        return self.calculate_star_formation_rate(z_t) * ft_d

    def calculate_SIa_rate(self, z, eta=0.04):
        """Calculates the rate of SN Ia. (Eq 15 - Oguri and Marshall 2010)

        :param z: redshift (z>=0)
        :param eta: canonical efficiency

        :return: SN Ia rate n(z) in [(h)yr^(-1)Mpc^(-3)]
        :return type: array-like
        """
        C_Ia = 0.032
        denominator = self._denominator
        SNIa_rate_list = []

        for i in z:
            t_z = self._cosmo.age(i).to_value()  # Time at given redshift z

            numerator = integrate.quad(
                self._numerator_integrand, 0.1, t_z - self._t_min, args=(t_z,)
            )
            SNIa_rate = eta * C_Ia * (numerator[0] / denominator[0])
            SNIa_rate_list.append(SNIa_rate)

        return np.array(SNIa_rate_list)

    def calculate_event_rate(self, z):
        """Calculates the SN Ia event rate.

        (Eq 15 - Oguri and Marshall 2010)
        """
        return self.calculate_SIa_rate(z)


class EventPopulation(object):
    """Class to select and calculate event population models."""

    def _init_(self, model, cosmo, z_max):
        self.model_name = model

        if model == "BNS":
            self._model = BNSMergerRate(cosmo=cosmo, z_max=z_max)

        elif model == "SNIa":
            self._model = SNIaRate(cosmo=cosmo, z_max=z_max)

        else:
            raise ValueError("model should be chosen from 'BNS' or 'SNIa")

    def calculate_event_rate(self, z):
        return self._model.calculate_event_rate(z)
