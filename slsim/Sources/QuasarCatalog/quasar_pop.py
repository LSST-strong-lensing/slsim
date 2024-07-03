from skypy.galaxies.redshift import redshifts_from_comoving_density
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy.table import Table
import os

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
        h: float = 0.70,
        zeta: float = 2.98,
        xi: float = 4.05,
        z_star: float = 1.60,
        alpha: float = -3.31,
        beta: float = -1.45,
        phi_star: float = 5.34e-6 * (0.70**3),
        cosmo: FlatLambdaCDM = None,
        sky_area: Quantity = None,
        noise: bool = True,
        redshifts: np.ndarray = None,
    ):
        """Initializes the QuasarRate class with given parameters.

        :param h: Hubble constant parameter H0/100, where H0 = 70 km s^-1 Mpc^-1.
        :type h: float
        :param zeta: (1) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survey (Richards et al. 2006: DOI: 10.1086/503559)
        :type zeta: float
        :param xi: (2) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survey (Richards et al. 2006: DOI: 10.1086/503559)
        :type xi: float
        :param z_star: (3) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survey (Richards et al. 2006: DOI: 10.1086/503559)
        :type z_star: float
        :param alpha: Bright end slope of quasar luminosity density profile.
        :type alpha: float
        :param beta: Faint end slope of quasar luminosity density profile.
        :type beta: float
        :param phi_star: Renormalization of the quasar luminosity function for a given h.
        :type phi_star: float
        :param cosmo: Cosmology object.
        :type cosmo: ~astropy.cosmology object
        :param sky_area: Sky area for sampled quasars in [solid angle].
        :type sky_area: `~Astropy.units.Quantity`
        :param noise: Poisson-sample the number of galaxies in quasar density lightcone.
        :type noise: bool
        :param redshifts: Redshifts for quasar density lightcone to be evaluated at.
        :type redshifts: np.ndarray
        """
        self.h = h
        self.zeta = zeta
        self.xi = xi
        self.z_star = z_star
        self.alpha = alpha
        self.beta = beta
        self.phi_star = phi_star
        self.cosmo = cosmo if cosmo is not None else FlatLambdaCDM(H0=70, Om0=0.3)
        self.sky_area = (
            sky_area if sky_area is not None else Quantity(0.05, unit="deg2")
        )
        self.noise = noise
        self.redshifts = (
            np.array(redshifts) if redshifts is not None else np.linspace(0.1, 5.0, 100)
        )

        # Construct the dynamic path to the data file
        base_path = os.path.dirname(os.path.abspath("__file__"))
        file_path = os.path.join(
            base_path, "data", "Quasar_K_Corrections", "i_band_Richards_et_al_2006.txt"
        )
        data = np.loadtxt(file_path)

        # The data is assumed to be in two columns: redshift and K-correction
        self.redshifts_kcorr = data[:, 0]
        self.K_corrections = data[:, 1]

        # Precompute the interpolation function
        self.K_corr_interp = interp1d(
            self.redshifts_kcorr,
            self.K_corrections,
            kind="linear",
            fill_value="extrapolate",
        )

    def M_star(self, z_value):
        """Calculates the break absolute magnitude of quasars for a given redshift
        according to Eq. (11) in Oguri & Marshall (2010): DOI:
        10.1111/j.1365-2966.2010.16639.x.

        :param z_value: Redshift value.
        :type z_value: float or np.ndarray
        :return: M_star value.
        :rtype: float or np.ndarray :unit: mag
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
        :type M: float or numpy.ndarray
        :param z_value: Redshift value.
        :type z_value: float or numpy.ndarray
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

    def convert_magnitude(self, magnitude, z, conversion="apparent_to_absolute"):
        """Converts between apparent and absolute magnitudes using K-corrections
        determined in Table 4 of Richards et al. 2006: DOI: 10.1086/503559.

        :param magnitude: Apparent or absolute i-band magnitude.
        :type magnitude: float or np.ndarray
        :param z: Redshift.
        :type z: float or np.ndarray
        :param conversion: Conversion direction, either 'apparent_to_absolute' or
            'absolute_to_apparent'.
        :type conversion: str
        :return: Converted magnitude.
        :rtype: float or np.ndarray :unit: mag
        """

        DM = self.cosmo.distmod(z).value
        K_corr = self.K_corr_interp(z)

        if conversion == "apparent_to_absolute":
            converted_magnitude = magnitude - DM - K_corr
        elif conversion == "absolute_to_apparent":
            converted_magnitude = magnitude + DM + K_corr
        else:
            raise ValueError(
                "Conversion must be either 'apparent_to_absolute' or 'absolute_to_apparent'"
            )

        return converted_magnitude

    def n_comoving(self, m_min, m_max, z_value):
        """Calculates the comoving number density of quasars for a given redshift by
        integrating dPhi/dM over the range of apparent magnitudes.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float or np.ndarray
        :param m_max: Maximum apparent magnitude.
        :type m_max: float or np.ndarray
        :param z_value: Redshift value.
        :type z_value: float or np.ndarray
        :return: Comoving number density of quasars.
        :rtype: float or np.ndarray :unit: Mpc^-3
        """
        M_min = self.convert_magnitude(
            m_min, z_value, conversion="apparent_to_absolute"
        )
        M_max = self.convert_magnitude(
            m_max, z_value, conversion="apparent_to_absolute"
        )

        if isinstance(z_value, np.ndarray):
            integrals = np.zeros_like(z_value)
            for i, z in enumerate(z_value):
                integral, _ = quad(self.dPhi_dM, M_min[i], M_max[i], args=(z,))
                integrals[i] = integral
            return integrals
        else:
            integral, _ = quad(self.dPhi_dM, M_min, M_max, args=(z_value,))
            return integral

    def generate_quasar_redshifts(self, m_min, m_max):
        """Generates redshift locations of quasars using a light cone formulation.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :return: Redshift locations of quasars.
        :rtype: np.ndarray
        """
        n_comoving_values = np.array(
            [self.n_comoving(m_min, m_max, z) for z in self.redshifts]
        )

        sampled_redshifts = redshifts_from_comoving_density(
            redshift=self.redshifts,
            density=n_comoving_values,
            sky_area=self.sky_area,
            cosmology=self.cosmo,
            noise=self.noise,
        )

        # Ensure redshifts are stored as numpy array
        sampled_redshifts = np.array(sampled_redshifts, dtype=float)

        return sampled_redshifts

    def compute_cdf_data(self, m_min, m_max, quasar_redshifts):
        """Computes cumulative distribution function (CDF) data for given redshift
        values.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :param quasar_redshifts: Redshift values generated from `generate_quasar_redshifts`.
        :type quasar_redshifts: array-like
        :return: Dictionary containing CDF data for each redshift.
        :rtype: dict
        """
        cdf_data_dict = {}

        for z in np.unique(quasar_redshifts):
            M_min = self.convert_magnitude(m_min, z, conversion="apparent_to_absolute")
            M_max = self.convert_magnitude(m_max, z, conversion="apparent_to_absolute")

            M_values = np.linspace(M_min, M_max, 100)

            dPhi_dM_values = np.array([self.dPhi_dM(M, z) for M in M_values])

            sorted_M_values = np.sort(M_values)
            cumulative_probabilities = np.cumsum(dPhi_dM_values)
            max_cumulative_probabilities = np.max(cumulative_probabilities)
            cumulative_prob_norm = (
                cumulative_probabilities / max_cumulative_probabilities
            )

            cdf_data_dict[z] = (sorted_M_values, cumulative_prob_norm)

        return cdf_data_dict

    def inverse_cdf_fits_for_redshifts(self, m_min, m_max, quasar_redshifts):
        """Creates inverse Cumulative Distribution Function (CDF) fits for each
        redshift.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :param quasar_redshifts: Redshift values generated from `generate_quasar_redshifts`.
        :type quasar_redshifts: array-like
        :return: Dictionary containing inverse CDF functions for each redshift.
        :rtype: dict
        """
        cdf_data = self.compute_cdf_data(m_min, m_max, quasar_redshifts)
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

    def quasar_sample(self, m_min, m_max, seed=42):
        """Generates random redshift values and associated apparent i-band magnitude
        values for quasar samples.

        :param m_min: Minimum apparent magnitude.
        :type m_min: float
        :param m_max: Maximum apparent magnitude.
        :type m_max: float
        :param seed: Random seed for reproducibility.
        :type seed: int
        :return: astropy Table with redshift and associated apparent i-band magnitude values.
        :rtype: `~astropy.table.Table`
        """
        np.random.seed(seed)
        quasar_redshifts = self.generate_quasar_redshifts(m_min=m_min, m_max=m_max)
        inverse_cdf_dict = self.inverse_cdf_fits_for_redshifts(
            m_min, m_max, quasar_redshifts
        )
        table_data = {"Redshift": [], "Apparent_i_mag": []}

        for redshift in quasar_redshifts:
            inverse_cdf = inverse_cdf_dict[redshift]
            random_inverse_cdf_value = np.random.rand()
            random_abs_M_value = inverse_cdf(random_inverse_cdf_value)

            # Convert the absolute magnitude back to apparent magnitude
            apparent_i_mag = self.convert_magnitude(
                random_abs_M_value, redshift, conversion="absolute_to_apparent"
            )

            table_data["Redshift"].append(redshift)
            table_data["Apparent_i_mag"].append(apparent_i_mag)

        # Create an Astropy Table from the collected data
        table = Table(table_data)
        return table
