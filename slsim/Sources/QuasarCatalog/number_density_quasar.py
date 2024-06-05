import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table
from matplotlib.lines import Line2D

""" 
References: 
Richards et al. 2005: DOI: 10.1111/j.1365-2966.2005.09096.x 
Richards et al. 2006: DOI: 10.1086/503559 
Oguri & Marshall (2010): DOI: 10.1111/j.1365-2966.2010.16639.x.
"""

def M_star(
        z_value, 
        h=0.72, 
        zeta=2.98, 
        xi=4.05, 
        z_star=1.60
):
    """ Calculates M_star for a given redshift according to Eq (11) in Oguri & Marshall (2010)

    Parameters:
    :param z_value: Redshift value.
    :type z_value: float
    :param h: Hubble constant parameter H0/100, where H0 = 72 km s^-1 Mpc^-1.
    :param zeta: (1) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survery (Richards et al. 2006)
    :param xi: (2) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survery (Richards et al. 2006)
    :param z_star: (3) Best fit value of the observed evolution of the quasar luminosity function from SDSS DR3 survery (Richards et al. 2006)
    :Note: You can adjust these values if needed 

    Returns:
    :rtype: float
    :return: M_star value.
    """

    denominator = (np.sqrt(np.exp(xi * z_value)) + np.sqrt(np.exp(xi * z_star)))**2
    if denominator <= 0:
        return np.nan  # or any other suitable value to indicate an error
    else:
        return -20.90 + (5 * np.log10(h)) - (2.5 * np.log10(np.exp(zeta * z_value) * (1 + np.exp(xi * z_star)) / denominator))

def dPhi_dM(
        M, 
        z_value, 
        alpha=-3.31, 
        beta=-1.45, 
        phi_star= 5.34e-6 * (0.72**3)
):
    """ Calculates dPhi_dM for a given M and redshift according to Eq (10) in Oguri & Marshall (2010).

    Parameters:
    :param M: Absolute i-band magnitude.
    :type M: float
    :param alpha: Bright end slope of quasar luminosity density profile obtained from SDSS DR3 and 2dF surveys (Richards et al. 2005) 
    :param beta: Faint end slope of quasar luminosity density profile obtained from SDSS DR3 and 2dF surveys (Richards et al. 2005)
    :param phi_star: Function to calculate the renormalization of the quasar luminosity function for a given h.

    Returns:
    :rtype: float
    :return: dPhi_dM value.
    """
   
    if z_value > 3:
        # Adjust the bright end slope for redshifts greater than 3 based on observations.
        alpha_val = -2.58
    else:
        # Use the default value for the bright end slope if redshift is 3 or lower.
        alpha_val = alpha

    M_star_value = M_star(z_value)

    denominator_dphi_dm = (10**(0.4 * (alpha_val + 1) * (M - M_star_value))) + (10**(0.4 * (beta + 1) * (M - M_star_value)))
    if denominator_dphi_dm == 0:
        return np.nan  # or any other suitable value to indicate an error

    term1 = phi_star / denominator_dphi_dm

    return term1

def cdf_fits_for_redshifts(
        M_values, random_redshift_values
):
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

    # Create an empty list to collect artists for the legend
    legend_handles_cdf = []

    # Plot the CDF vs. M curve for each randomly generated redshift
    for random_redshift in random_redshift_values:
        dPhi_dM_random_values = [dPhi_dM(M, random_redshift) for M in M_values]
        sorted_M_values = np.sort(M_values) 
        cumulative_probabilities = np.cumsum(dPhi_dM_random_values)
        cumulative_prob_norm = cumulative_probabilities / max(cumulative_probabilities)
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

    inverse_cdf_dict = {}

    for random_redshift in random_redshift_values:
        # Calculate cumulative probability
        dPhi_dM_values = [dPhi_dM(M, random_redshift) for M in M_values]
        cumulative_prob = np.cumsum(dPhi_dM_values) / np.sum(dPhi_dM_values)

        # Interpolate the inverse CDF function
        inverse_cdf = interp1d(cumulative_prob, np.sort(M_values), kind='linear', fill_value='extrapolate')

        # Store the inverse CDF function in the dictionary
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

    # Print the table
    print(table)

    return table


# Overall test code usuage
def main():
    np.random.seed(42)
    random_redshift = int(input("Enter the number of redshift values to generate: "))
    random_redshift_values = np.random.uniform(0, 5, random_redshift)
    M_values = np.linspace(-28, -24, 100)

    # Plot dPhi/dM vs. M for each redshift
    plt.figure(figsize=(10, 6))
    for z in random_redshift_values:
        dphi_dm_values = [dPhi_dM(M, z) for M in M_values]
        plt.plot(M_values, dphi_dm_values, label=f'z={z}')

    plt.xlabel('Absolute i-band Magnitude (M)')
    plt.ylabel(r'$\frac{d\Phi_{QSO}}{dM}$ [mag$^{-1}$ Mpc$^{-3}$]')
    plt.title(r'$\frac{d\Phi_{QSO}}{dM}$ vs. M')
    plt.xlim(-24, -28)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the CDFs for each redshift
    plt.figure(figsize=(10, 6))
    legend_handles_cdf = cdf_fits_for_redshifts(M_values, random_redshift_values)
    plt.xlabel('Absolute i-band Magnitude (M)')
    plt.ylabel('CDF')
    plt.title('CDF vs. M')
    plt.legend(handles=legend_handles_cdf)
    plt.grid(True)
    plt.show()

    # Plot the inverse CDFs for each redshift
    plt.figure(figsize=(10, 6))
    inverse_cdf_legend_handles = []
    inverse_cdf_dict = inverse_cdf_fits_for_redshifts(M_values, random_redshift_values)
    for z in random_redshift_values:
        cdf_values = np.linspace(0, 1, 1000)
        M_values_given_cdf = inverse_cdf_dict[z](cdf_values)
        plt.plot(cdf_values, M_values_given_cdf, label=f'z={z}')
        inverse_cdf_legend_handles.append(Line2D([0], [0], color='b', linewidth=2, label=f'z={z}'))

    plt.xlabel('Inverse CDF')
    plt.ylabel('Absolute i-band Magnitude M')
    plt.title('Inverse CDF vs. M')
    plt.legend(handles=inverse_cdf_legend_handles)
    plt.grid(True)
    plt.show()

    # Generate the table with random redshift values
    table = generate_redshift_table(random_redshift_values, inverse_cdf_dict)

if __name__ == "__main__":
    main()