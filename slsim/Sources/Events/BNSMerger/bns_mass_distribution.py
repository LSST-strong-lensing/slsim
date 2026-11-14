import numpy as np
from scipy.stats import truncnorm

"""References:
BNS mass distribution: Farrow, Zhu and Thrane 2019.
"""


class BNSMassDistribution(object):
    """Assign neutron star(NS) masses to each event and calculate chirp mass
    and mass ratio.

    All mass parameters are in solar mass.
    """

    def __init__(self):
        self._mu = np.array(
            [1.34, 1.47]
        )  # Means of the two Gaussian components for NS masses
        self._sigma = np.array(
            [0.02, 0.15]
        )  # Standard deviations of the two Gaussian components
        self._weight_1 = 0.68  # Weight of the first Gaussian component (dimensionless)

        # Bound for recycled neutron star masses which is the first-born neutron star
        self._mr_min = 0.8
        self._mr_max = 2.0

        # Bound for slow neutron star which is the second-born neutron star
        self._ms_min = 1.16
        self._ms_max = 1.42

    def assign_masses(self, population):
        """Assign independently sampled recycled and slow NS masses.

        Distribution parameters are fixed to the posterior medians
        reported by Farrow, Zhu and Thrane (2019).

        :param population: event population as an astropy Table
        :return: a copy with component masses and chirp mass in solar
            masses, and dimensionless mass ratio q <= 1
        """
        size = len(population)

        # Select a Gaussian component for each recycled NS.
        component = np.where(np.random.random(size) < self._weight_1, 0, 1)
        mu = self._mu[component]
        sigma = self._sigma[component]

        # Convert mass bounds to standardized truncnorm bounds.
        lower = (self._mr_min - mu) / sigma
        upper = (self._mr_max - mu) / sigma

        mr = truncnorm.rvs(lower, upper, loc=mu, scale=sigma, size=size)
        ms = np.random.uniform(self._ms_min, self._ms_max, size=size)

        m1 = np.maximum(mr, ms)
        m2 = np.minimum(mr, ms)

        result = population.copy()
        result["mass_recycled"] = mr
        result["mass_slow"] = ms
        result["mass_1"] = m1
        result["mass_2"] = m2
        result["Mchirp"] = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
        result["q"] = m2 / m1

        return result
