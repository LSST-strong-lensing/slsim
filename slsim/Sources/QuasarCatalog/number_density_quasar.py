import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table

def M_star(
        z_value, 
        h=0.72, 
        zeta=2.98, 
        xi=4.05, 
        z_star=1.60
):
    """ Calculates M_star for a given redshift according to Eq (11) in Oguri & Marshall (2010).

    Parameters:
    :param z_value: Redshift value.
    :type z_value: float
    :param h: Hubble constant parameter H0/100, where H0 = 72 km s^-1 Mpc^-1.
    :param zeta: Best fit value of the observed evolution of the quasar luminosity function (1)
    :param xi: Best fit value of the observed evolution of the quasar luminosity function (2) 
    :param z_star: Best fit value of the observed evolution of the quasar luminosity function (3)
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
    :param phi_star: Function to calculate the renormalization of the quasar luminosity function for given h.

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
        M_values, 
        random_redshift_values
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
        line, = plt.plot(sorted_M_values, cumulative_prob_norm, label=f'Random Redshift: {random_redshift}')
        legend_handles_cdf.append(line)
      
    return legend_handles_cdf

def inverse_cdf_fits_for_redshifts(
        M_values, 
        random_redshift_values
):
    """Creates inverse Cumulative Distribution Function (CDF) plots for each randomly generated redshift.

    Parameters:
    :param M_values: Array of absolute i-band magnitudes.
    :type M_values: numpy.ndarray

    :param random_redshift_values: Array of randomly generated redshift values in previous cell. 
    :type random_redshift_values: numpy.ndarray

    Returns:
    :rtype: list
    :return: List of legend handles for each inverse CDF plot.
    """

    legend_handles_inverse_cdf = []

    for random_redshift in random_redshift_values:

        # Calculate cumulative probability
        dPhi_dM_values = [dPhi_dM(M, random_redshift) for M in M_values]
        cumulative_prob = np.cumsum(dPhi_dM_values) / np.sum(dPhi_dM_values)

        # Interpolate the inverse CDF function
        inverse_cdf = interp1d(cumulative_prob, np.sort(M_values), kind='linear', fill_value='extrapolate')

        # Generate M(cdf) function
        def M_given_cdf(cdf_value):
            return inverse_cdf(cdf_value)

        # Define cdf_values for plotting
        cdf_values = np.linspace(0, 1, 1000)

        # Plot the inverse CDF function
        M_values_given_cdf = inverse_cdf(cdf_values)
        line, = plt.plot(cdf_values, M_values_given_cdf, label=f'M(CDF) for z = {random_redshift:.2f}')
        legend_handles_inverse_cdf.append(line)

    return legend_handles_inverse_cdf

def generate_redshift_table(
        random_redshift_values, 
        inverse_cdf_dict, seed=42
):
    """Generates random redshift values and associated M values using inverse CDF.

    Parameters:
    :param random_redshift_values: Array of randomly generated redshift values.
    :type random_redshift_values: list or numpy.ndarray

    :param inverse_cdf_dict: Dictionary of inverse CDF functions for each redshift.
    :type inverse_cdf_dict: dict

    :param seed: Seed for reproducibility (default: 42).
    :type seed: int

    Returns:
    :rtype: astropy.table.Table
    :return: Astropy table containing the generated random redshift values and associated M values.
    """
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Create an empty list to collect data for the table
    table_data = {'Associated_M': [], 'Redshift': []}

    # Generate random CDF value and associated M value for each redshift
    for random_redshift in random_redshift_values:
        
        # Generate a random CDF value between 0 and 1
        random_cdf_value = np.random.rand()
        
        # Get the associated M value using the inverse CDF function
        inverse_cdf = inverse_cdf_dict[random_redshift]
        random_M_value = inverse_cdf(random_cdf_value)
        
        # Append the data to the table
        table_data['Associated_M'].append(random_M_value)
        table_data['Redshift'].append(random_redshift)

    # Create the Astropy table
    table = Table(table_data)

    # Print the table
    print(table)

    return table

# Overall example usage 

def main():
    np.random.seed(42)
    num_redshifts = int(input("Enter the number of redshift values to generate: "))
    random_redshift_values = np.random.uniform(0, 5, num_redshifts)
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

    plt.figure(figsize=(10, 6))
    legend_handles_cdf = cdf_fits_for_redshifts(M_values, random_redshift_values)
    plt.xlabel('Absolute i-band Magnitude (M)')
    plt.ylabel('CDF')
    plt.title('CDF vs. M')
    plt.legend(handles=legend_handles_cdf)
    plt.grid(True)
    plt.show()

    # Plot the inverse CDFs
    plt.figure(figsize=(10, 6))
    inverse_cdf_legend_handles = inverse_cdf_fits_for_redshifts(M_values, random_redshift_values)
    plt.xlabel('Cumulative Distribution Function (CDF)')
    plt.ylabel('Absolute i-band Magnitude M')
    plt.title('Inverse CDF vs. M for Random Redshifts')
    plt.legend(handles=inverse_cdf_legend_handles)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()