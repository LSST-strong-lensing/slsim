from slsim.Sources.QuasarCatalog.number_density_quasar import QuasarRate
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from astropy.table import Table
import pytest


class TestQuasarRate:
    """Class to test the QuasarRate class."""

    def setup_method(self):
        """Setup the QuasarRate instance before each test method."""
        self.quasar_rate = QuasarRate(
            h=0.70,
            zeta=2.98,
            xi=4.05,
            z_star=1.60,
            alpha=-3.31,
            beta=-1.45,
            phi_star=5.34e-6 * (0.70**3),
        )  # Ensure parameters match those used in tests

    def test_M_star(self):
        # Test case 1: Regular case
        z_value = 2.01
        expected_value = -25.59096969
        m_star_calc = self.quasar_rate.M_star(z_value)
        np.testing.assert_almost_equal(m_star_calc, expected_value, decimal=3)

        # Test case 2: Edge case where the denominator should be zero
        self.quasar_rate.xi = (
            -1000
        )  # Using a large negative value for xi to ensure the denominator is zero
        with pytest.raises(
            ValueError,
            match="Encountered zero denominator in M_star calculation. Check input values.",
        ):
            self.quasar_rate.M_star(z_value)

    def test_dPhi_dM(self):
        M = -28.0
        alpha = -3.31
        beta = -1.45

        # Test case 1: z_value <= 3 (z_value = 2.01)
        z_value = 2.01
        M_star_value = self.quasar_rate.M_star(z_value)
        expected_denominator = (10 ** (0.4 * (alpha + 1) * (M - M_star_value))) + (
            10 ** (0.4 * (beta + 1) * (M - M_star_value))
        )
        if expected_denominator == 0:
            expected_value = np.nan
        else:
            expected_value = 1.071374609e-8
        dphi_dm_calc = self.quasar_rate.dPhi_dM(M, z_value)
        if np.isnan(expected_value):
            assert np.isnan(dphi_dm_calc), f"Expected np.nan but got {dphi_dm_calc}"
        else:
            np.testing.assert_almost_equal(dphi_dm_calc, expected_value, decimal=3)

        # Test case 2: z_value > 3 (z_value = 5)
        z_value = 5
        alpha_val = -2.58
        M_star_value = self.quasar_rate.M_star(z_value)
        expected_denominator = (10 ** (0.4 * (alpha_val + 1) * (M - M_star_value))) + (
            10 ** (0.4 * (beta + 1) * (M - M_star_value))
        )
        if expected_denominator == 0:
            expected_value = np.nan
        else:
            expected_value = 1.091480781e-9
        dphi_dm_calc = self.quasar_rate.dPhi_dM(M, z_value)
        if np.isnan(expected_value):
            assert np.isnan(dphi_dm_calc), f"Expected np.nan but got {dphi_dm_calc}"
        else:
            np.testing.assert_almost_equal(dphi_dm_calc, expected_value, decimal=3)

    def test_compute_cdf_data(self):
        # Generate test data
        M_values = np.linspace(-30, -20, 100)  # Example range of magnitudes
        random_redshift_values = np.linspace(0.5, 2.5, 5)  # Example redshift values

        # Call the function
        cdf_data = self.quasar_rate.compute_cdf_data(M_values, random_redshift_values)

        # Verify the output format
        assert isinstance(cdf_data, dict), "Output should be a dictionary"
        assert set(cdf_data.keys()) == set(
            random_redshift_values
        ), "Keys should match the input redshift values"

        for redshift, (sorted_M_values, cumulative_prob_norm) in cdf_data.items():
            # Check that sorted_M_values is a sorted version of M_values
            assert np.array_equal(
                sorted_M_values, np.sort(M_values)
            ), "M values should be sorted"

            # Check that cumulative_prob_norm is a proper cumulative distribution
            assert np.all(cumulative_prob_norm >= 0) and np.all(
                cumulative_prob_norm <= 1
            ), "Cumulative probabilities should be between 0 and 1"
            assert np.isclose(
                cumulative_prob_norm[-1], 1.0
            ), "Cumulative probabilities should end at 1"

            # Check that the length of the arrays match
            assert len(sorted_M_values) == len(
                M_values
            ), "Sorted M values should have the same length as input M values"
            assert len(cumulative_prob_norm) == len(
                M_values
            ), "Cumulative probabilities should have the same length as input M values"

    def test_inverse_cdf_fits_for_redshifts(self):
        # Generate some random redshift values and M values
        np.random.seed(42)
        random_redshift_values = np.random.uniform(0.1, 5.0, size=5)
        M_values = np.random.uniform(-30, -20, size=1000)

        # Generate inverse CDF functions for the random redshifts
        inverse_cdf_dict = self.quasar_rate.inverse_cdf_fits_for_redshifts(
            M_values, random_redshift_values
        )

        # Draw samples from each inverse CDF and create histograms
        num_samples = 10000
        sample_results = {}
        for z_value, inverse_cdf_func in inverse_cdf_dict.items():
            samples = inverse_cdf_func(np.random.rand(num_samples))
            sample_results[z_value] = samples

        # Plot histograms for each redshift and compare with CDF plots
        fig, axs = plt.subplots(len(random_redshift_values), 2, figsize=(12, 8))
        fig.tight_layout(pad=3.0)

        for i, z_value in enumerate(random_redshift_values):
            # Plot the histogram
            axs[i, 0].hist(sample_results[z_value], bins=30, density=True, alpha=0.7)
            axs[i, 0].set_title(f"Histogram of samples for z={z_value}")

            # Plot the CDF
            M_values_sorted = np.sort(M_values)
            cumulative_probabilities = np.cumsum(
                [self.quasar_rate.dPhi_dM(M, z_value) for M in M_values_sorted]
            )
            cumulative_prob_norm = cumulative_probabilities / max(
                cumulative_probabilities
            )
            axs[i, 1].plot(
                M_values_sorted,
                cumulative_prob_norm,
                label=f"CDF for z={z_value}",
                color="r",
            )
            axs[i, 1].legend(loc="lower right")
            axs[i, 1].set_title(f"CDF plot for z={z_value}")

            # Perform KS test to compare histogram with CDF
            _, p_value = ks_2samp(sample_results[z_value], cumulative_prob_norm)
            assert (
                p_value < 0.05
            ), f"KS Test failed for z={z_value}: p-value = {p_value}"

    def test_quasar_sample(self):
        # Generate some random redshift values and inverse CDF functions
        np.random.seed(42)
        random_redshift_values = np.random.uniform(0.1, 5.0, size=5)
        M_values = np.random.uniform(-30, -20, size=1000)
        inverse_cdf_dict = self.quasar_rate.inverse_cdf_fits_for_redshifts(
            M_values, random_redshift_values
        )

        # Call the function under test
        table = self.quasar_rate.quasar_sample(random_redshift_values, M_values)

        # Assertion: Check if the returned value is an Astropy table
        assert isinstance(table, Table), "Generated table is not an Astropy Table"


# Running the tests with pytest
if __name__ == "__main__":
    pytest.main()
