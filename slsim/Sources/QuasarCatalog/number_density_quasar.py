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

def M_star(z_value, h=0.72, zeta=2.98, xi=4.05, z_star=1.60):
    """Calculates the break absolute magnitude of quasars for a given redshift according
    to Eq. (11) in Oguri & Marshall (2010): DOI: 10.1111/j.1365-2966.2010.16639.x.

    Parameters:
    :param z_value: Redshift value.
    :type z_value: float or np.ndarray
    :param h: Hubble constant parameter H0/100, where H0 = 72 km s^-1 Mpc^-1.
    :param zeta: (1) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survery (Richards et al. 2006: DOI: 10.1086/503559)
    :param xi: (2) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survery (Richards et al. 2006: DOI: 10.1086/503559)
    :param z_star: (3) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survery (Richards et al. 2006: DOI: 10.1086/503559)
    :Note: You can adjust these values if needed

    Returns:
    :rtype: float or np.ndarray
    :return: M_star value.
    :unit: magnitudes (mag)
    """
    # Convert z_value to a numpy array if it isn't already
    z_value = np.atleast_1d(z_value)
    
    denominator = (np.sqrt(np.exp(xi * z_value)) + np.sqrt(np.exp(xi * z_star))) ** 2
    result = (
        -20.90
        + (5 * np.log10(h))
        - (
            2.5
            * np.log10(
                np.exp(zeta * z_value) * (1 + np.exp(xi * z_star)) / denominator
            )
        )
    )
    
    # Handle zero denominator cases
    result[denominator == 0] = np.nan
    
    return result if result.size > 1 else result.item()

def dPhi_dM(M, z_value, alpha=-3.31, beta=-1.45, phi_star=5.34e-6 * (0.72**3)):
    """Calculates dPhi_dM for a given M and redshift according to Eq (10) in Oguri &
    Marshall (2010): DOI: 10.1111/j.1365-2966.2010.16639.x..

    Parameters:
    :param M: Absolute i-band magnitude.
    :type M: float or np.ndarray
    :param alpha: Bright end slope of quasar luminosity density profile obtained from SDSS DR3 and 2dF surveys (Richards et al. 2005: DOI: 10.1111/j.1365-2966.2005.09096.x )
    :param beta: Faint end slope of quasar luminosity density profile obtained from SDSS DR3 and 2dF surveys (Richards et al. 2005: DOI: 10.1111/j.1365-2966.2005.09096.x )
    :param phi_star: Function to calculate the renormalization of the quasar luminosity function for a given h.

    Returns:
    :rtype: float or np.ndarray
    :return: dPhi_dM value.
    :unit: mag^-1 Mpc^-3
    """
    # Convert inputs to numpy arrays if they are not already
    M = np.atleast_1d(M)
    z_value = np.atleast_1d(z_value)
    
    if z_value.shape == ():
        z_value = np.full_like(M, z_value)
    if M.shape == ():
        M = np.full_like(z_value, M)
    
    # Adjust the bright end slope for redshifts greater than 3 based on observations.
    alpha_val = np.where(z_value > 3, -2.58, alpha)

    # Calculate M_star values for all z_values
    M_star_value = M_star(z_value)

    denominator_dphi_dm = (10 ** (0.4 * (alpha_val + 1) * (M - M_star_value))) + (
        10 ** (0.4 * (beta + 1) * (M - M_star_value))
    )
    term1 = np.divide(phi_star, denominator_dphi_dm, out=np.full_like(denominator_dphi_dm, np.nan), where=denominator_dphi_dm != 0)
    
    return term1 if term1.size > 1 else term1.item()

def compute_cdf_data(M_values, random_redshift_values):
    """Computes the CDF data for each randomly generated redshift.

    Parameters:
    :param M_values: Array of absolute i-band magnitudes.
    :type M_values: numpy.ndarray
    :param random_redshift_values: Array of randomly generated redshift values.
    :type random_redshift_values: numpy.ndarray

    Returns:
    :rtype: dict
    :return: Dictionary containing the sorted M values and corresponding cumulative probabilities for each redshift.
    """
    cdf_data_dict = {}

    for random_redshift in random_redshift_values:
        dPhi_dM_values = [dPhi_dM(M, random_redshift) for M in M_values]
        sorted_M_values = np.sort(M_values)
        cumulative_probabilities = np.cumsum(dPhi_dM_values)
        cumulative_prob_norm = cumulative_probabilities / max(cumulative_probabilities)
        cdf_data_dict[random_redshift] = (sorted_M_values, cumulative_prob_norm)

    return cdf_data_dict

def cdf_fits_for_redshifts(M_values, random_redshift_values):
    """Creates Cumulative Distribution Functions (CDF) plots for each randomly generated redshift.

    Parameters:
    :param M_values: Array of absolute i-band magnitudes.
    :type M_values: numpy.ndarray
    :param random_redshift_values: Array of randomly generated redshift values.
    :type random_redshift_values: numpy.ndarray

    Returns:
    :rtype: list
    :return: List of legend handles for each CDF plot.
    """
    # Compute the CDF data
    cdf_data = compute_cdf_data(M_values, random_redshift_values)

    # Create an empty list to collect artists for the legend
    legend_handles_cdf = []

    # Plot the CDF vs. M curve for each randomly generated redshift
    for random_redshift, (sorted_M_values, cumulative_prob_norm) in cdf_data.items():
        line, = plt.plot(sorted_M_values, cumulative_prob_norm, label=f'z={random_redshift}')
        legend_handles_cdf.append(line)

    return legend_handles_cdf

def inverse_cdf_fits_for_redshifts(M_values, random_redshift_values):
    """Creates inverse Cumulative Distribution Function (CDF) fits for each randomly generated redshift.

    Parameters:
    :param M_values: Array of absolute i-band magnitudes.
    :type M_values: numpy.ndarray
    :param random_redshift_values: Array of randomly generated redshift values.
    :type random_redshift_values: numpy.ndarray

    Returns:
    :rtype: dict
    :return: Dictionary of inverse CDF functions for each redshift.
    """
    # Compute the CDF data
    cdf_data = compute_cdf_data(M_values, random_redshift_values)

    inverse_cdf_dict = {}

    for random_redshift, (sorted_M_values, cumulative_prob_norm) in cdf_data.items():
        # Interpolate the inverse CDF function
        inverse_cdf = interp1d(cumulative_prob_norm, sorted_M_values, kind='linear', fill_value='extrapolate')
        inverse_cdf_dict[random_redshift] = inverse_cdf

    return inverse_cdf_dict

def generate_redshift_table(random_redshift_values, inverse_cdf_dict, seed=42):
    """Generates random redshift values, associated inverse CDF values, and M values.

    Parameters:
    :param random_redshift_values: Array of randomly generated redshift values.
    :type random_redshift_values: list or numpy.ndarray

    :param inverse_cdf_dict: Dictionary of inverse CDF functions for each redshift.
    :type inverse_cdf_dict: dict

    :param seed: Seed for reproducibility (default: 42).
    :type seed: int

    Returns:
    :rtype: astropy.table.Table
    :return: Astropy table containing the generated data (Redshift, Inverse CDF Value, Associated M).
    """
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Create an empty list to collect data for the table
    table_data = {'Redshift': [], 'Inverse_CDF_Value': [], 'Associated_M': []}

    # Generate random Inverse CDF value and associated M value for each redshift
    for random_redshift in random_redshift_values:
        # Find the inverse CDF function for the current random redshift
        inverse_cdf = inverse_cdf_dict[random_redshift]

        # Generate a random Inverse CDF value between 0 and 1
        random_inverse_cdf_value = np.random.rand()

        # Get the associated M value using the inverse CDF function
        random_M_value = inverse_cdf(random_inverse_cdf_value)

        # Append the data to the table
        table_data['Redshift'].append(random_redshift)
        table_data['Inverse_CDF_Value'].append(random_inverse_cdf_value)
        table_data['Associated_M'].append(random_M_value)

    # Create the Astropy table
    table = Table(table_data)

    return table