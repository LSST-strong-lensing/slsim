import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table

""" 
References: 
Richards et al. 2005
Richards et al. 2006
Oguri & Marshall (2010)
"""


class QuasarRate(object):
    """Class to calculate quasar luminosity functions and generate quasar samples."""

    def __init__(
        self,
        h=0.70,
        zeta=2.98,
        xi=4.05,
        z_star=1.60,
        alpha=-3.31,
        beta=-1.45,
        phi_star=5.34e-6 * (0.70**3),
    ):
        """Initializes the QuasarRate class with given parameters.

        :param h: Hubble constant parameter H0/100, where H0 = 70 km s^-1 Mpc^-1.
        :param zeta: (1) Best fit value of the observed evolution of the quasar
            luminosity function from SDSS DR3 survery (Richards et al. 2006: DOI:
            10.1086/503559)
        :param xi: (2) Best fit value of the observed evolution of the quasar luminosity
            function from SDSS DR3 survery (Richards et al. 2006: DOI: 10.1086/503559)
        :param z_star: (3) Best fit value of the observed evolution of the quasar
            luminosity function from SDSS DR3 survery (Richards et al. 2006: DOI:
            10.1086/503559)
        :param alpha: Bright end slope of quasar luminosity density profile.
        :param beta: Faint end slope of quasar luminosity density profile.
        :param phi_star: Renormalization of the quasar luminosity function for a given
            h.
        """
        self.h = h
        self.zeta = zeta
        self.xi = xi
        self.z_star = z_star
        self.alpha = alpha
        self.beta = beta
        self.phi_star = phi_star

    def M_star(self, z_value):
        """Calculates the break absolute magnitude of quasars for a given redshift
        according to Eq. (11) in Oguri & Marshall (2010): DOI:
        10.1111/j.1365-2966.2010.16639.x.

        :param z_value: Redshift value.
        :type z_value: float or np.ndarray
        :return: M_star value.
        :rtype: float or np.ndarray :unit: magnitudes (mag)
        """
        z_value = np.atleast_1d(z_value)
        denominator = (
            np.sqrt(np.exp(self.xi * z_value)) + np.sqrt(np.exp(self.xi * self.z_star))
        ) ** 2
        result = (
            -20.90
            + (5 * np.log10(self.h))
            - (
                2.5
                * np.log10(
                    np.exp(self.zeta * z_value)
                    * (1 + np.exp(self.xi * self.z_star))
                    / denominator
                )
            )
        )

        if np.any(denominator == 0):
            raise ValueError(
                "Encountered zero denominator in M_star calculation. Check input values."
            )

        return result

    def dPhi_dM(self, M, z_value):
        """Calculates dPhi_dM for a given M and redshift according to Eq (10) in Oguri &
        Marshall (2010): DOI: 10.1111/j.1365-2966.2010.16639.x.

        :param M: Absolute i-band magnitude.
        :type M: float or np.ndarray
        :param z_value: Redshift value.
        :type z_value: float or np.ndarray
        :return: dPhi_dM value.
        :rtype: float or np.ndarray :unit: mag^-1 Mpc^-3
        """
        M = np.atleast_1d(M)
        z_value = np.atleast_1d(z_value)

        if z_value.shape == ():
            z_value = np.full_like(M, z_value)
        if M.shape == ():
            M = np.full_like(z_value, M)

        alpha_val = np.where(z_value > 3, -2.58, self.alpha)
        M_star_value = self.M_star(z_value)

        denominator_dphi_dm = (10 ** (0.4 * (alpha_val + 1) * (M - M_star_value))) + (
            10 ** (0.4 * (self.beta + 1) * (M - M_star_value))
        )

        # Handle division by zero
        term1 = np.divide(
            self.phi_star,
            denominator_dphi_dm,
            out=np.full_like(denominator_dphi_dm, np.nan),
            where=denominator_dphi_dm != 0,
        )

        return term1

    def compute_cdf_data(self, M_values, redshift_values):
        """Computes the CDF data for each redshift.

        :param M_values: Array of absolute i-band magnitudes.
        :type M_values: numpy.ndarray
        :param redshift_values: Array of redshift values.
        :type redshift_values: numpy.ndarray
        :return: Dictionary containing the sorted M values and corresponding cumulative
            probabilities for each redshift.
        :rtype: dict
        """
        cdf_data_dict = {}

        dPhi_dM_values = [self.dPhi_dM(M, redshift_values) for M in M_values]
        sorted_M_values = np.sort(M_values)
        cumulative_probabilities = np.cumsum(dPhi_dM_values, axis=0)
        max_cumulative_probabilities = np.max(cumulative_probabilities, axis=0)
        cumulative_prob_norm = cumulative_probabilities / max_cumulative_probabilities
        for i, redshift in enumerate(redshift_values):
            cdf_data_dict[redshift] = (sorted_M_values, cumulative_prob_norm[:, i])

        return cdf_data_dict

    def inverse_cdf_fits_for_redshifts(self, M_values, redshift_values):
        """Creates inverse Cumulative Distribution Function (CDF) fits for each
        redshift.

        :param M_values: Array of absolute i-band magnitudes.
        :type M_values: numpy.ndarray
        :param redshift_values: Array of redshift values.
        :type redshift_values: numpy.ndarray
        :return: Dictionary of inverse CDF functions for each redshift.
        :rtype: dict
        """
        cdf_data = self.compute_cdf_data(M_values, redshift_values)

        inverse_cdf_dict = {}
        for redshift, (sorted_M_values, cumulative_prob_norm) in cdf_data.items():
            inverse_cdf = interp1d(
                cumulative_prob_norm,
                sorted_M_values,
                kind="linear",
                fill_value="extrapolate",
            )
            inverse_cdf_dict[redshift] = inverse_cdf

        return inverse_cdf_dict

    def quasar_sample(self, redshift_values, M_values, seed=42):
        """Generates random redshift values and associated M values for quasar samples.

        :param redshift_values: Array of redshift values.
        :type redshift_values: list or numpy.ndarray
        :param M_values: Array of absolute i-band magnitudes.
        :type M_values: numpy.ndarray
        :param seed: Seed for reproducibility (default: 42).
        :type seed: int
        :return: Astropy table containing the generated data (Redshift, Associated M).
        :rtype: astropy.table.Table
        """
        np.random.seed(seed)
        inverse_cdf_dict = self.inverse_cdf_fits_for_redshifts(
            M_values, redshift_values
        )
        table_data = {"Redshift": [], "Associated_M": []}

        for redshift in redshift_values:
            inverse_cdf = inverse_cdf_dict[redshift]
            random_inverse_cdf_value = np.random.rand()
            random_M_value = inverse_cdf(random_inverse_cdf_value)
            table_data["Redshift"].append(redshift)
            table_data["Associated_M"].append(random_M_value)

        table = Table(table_data)
        return table
