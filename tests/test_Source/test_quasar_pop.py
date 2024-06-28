from slsim.Sources.QuasarCatalog.quasar_pop import QuasarRate
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
import numpy as np
from scipy.stats import ks_2samp
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
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
            sky_area=Quantity(0.05, unit="deg2"),
            noise=True,
            redshifts=np.linspace(0.1, 5.0, 100),
        )

    def test_M_star(self):
        # Test case 1: Regular case
        z_value = 2.01
        expected_value = -25.59096969
        m_star_calc = self.quasar_rate.M_star(z_value)
        np.testing.assert_almost_equal(m_star_calc, expected_value, decimal=4)

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
            np.testing.assert_almost_equal(dphi_dm_calc, expected_value, decimal=4)

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
            np.testing.assert_almost_equal(dphi_dm_calc, expected_value, decimal=4)

    def test_convert_magnitude(self):
        # Test data: Example numbers taken directly from Table 5 of Richards et al. 2006: DOI: 10.1086/503559
        test_redshifts = [1.199, 2.240, 0.460, 0.949, 0.989]
        test_magnitudes = [19.08, 18.18, 19.09, 19.05, 18.99]

        expected_abs_mags = [
            -25.40439324,
            -27.86579516,
            -23.21158492,
            -25.02167014,
            -25.15264705,
        ]
        expected_app_mags = test_magnitudes

        # Test case #1: Apparent to Absolute Magnitude Conversion
        for z, mag, expected_abs_mag in zip(
            test_redshifts, test_magnitudes, expected_abs_mags
        ):
            abs_mag = self.quasar_rate.convert_magnitude(
                mag, z, conversion="apparent_to_absolute"
            )
            np.testing.assert_almost_equal(abs_mag, expected_abs_mag, decimal=4)

        # Test case #2: Absolute to Apparent Magnitude Conversion
        for z, expected_abs_mag, expected_app_mag in zip(
            test_redshifts, expected_abs_mags, expected_app_mags
        ):
            app_mag = self.quasar_rate.convert_magnitude(
                expected_abs_mag, z, conversion="absolute_to_apparent"
            )
            np.testing.assert_almost_equal(app_mag, expected_app_mag, decimal=4)

    def test_n_comoving(self):
        # Test data
        m_min = 15
        m_max = 25
        test_redshifts = [1.199, 3.151]

        # Test case #1: z_value <= 3
        # Test case 2: z_value > 3 (to account for the change in alpha in 'dPhi_dM()')
        expected_n_comoving = [5.294255979e-5, 1.077342851e-5]

        for z, expected_n in zip(test_redshifts, expected_n_comoving):
            n_comoving = self.quasar_rate.n_comoving(m_min, m_max, z)
            np.testing.assert_almost_equal(n_comoving, expected_n, decimal=4)

    def test_generate_quasar_redshifts(self):
        np.random.seed(42)

        sampled_redshifts = self.quasar_rate.generate_quasar_redshifts(
            m_min=15, m_max=25
        )
        assert isinstance(
            sampled_redshifts, np.ndarray
        ), f"Returned object is not a numpy array, got {type(sampled_redshifts)} instead."

        single_redshift = self.quasar_rate.generate_quasar_redshifts(
            m_min=15, m_max=25, return_single=True
        )
        assert isinstance(
            single_redshift, float
        ), f"Returned object is not a float, got {type(single_redshift)} instead."

        assert sampled_redshifts.size > 0, "No redshifts were generated."

    def test_compute_cdf_data(self):
        sampled_redshifts = self.quasar_rate.generate_quasar_redshifts(
            m_min=15, m_max=25
        )
        cdf_data_dict = self.quasar_rate.compute_cdf_data(
            m_min=15, m_max=25, quasar_redshifts=sampled_redshifts
        )

        assert isinstance(cdf_data_dict, dict), "Output should be a dictionary."

        computed_redshifts = list(cdf_data_dict.keys())
        assert np.allclose(
            computed_redshifts, np.unique(sampled_redshifts)
        ), "Keys should match the redshift values."

        # Check properties of each CDF data
        for z, (sorted_M_values, cumulative_prob_norm) in cdf_data_dict.items():
            assert np.all(np.diff(sorted_M_values) >= 0), "M values should be sorted."

            assert np.all(
                (cumulative_prob_norm >= 0) & (cumulative_prob_norm <= 1)
            ), "Cumulative probabilities should be between 0 and 1."

            assert np.isclose(
                cumulative_prob_norm[-1], 1.0
            ), "Cumulative probabilities should end at 1."

            assert len(sorted_M_values) == len(
                cumulative_prob_norm
            ), "Lengths of sorted M values and cumulative probabilities should match."

    def test_inverse_cdf_fits_for_redshifts_ks(self):
        # Test data
        np.random.seed(42)
        m_min = 15
        m_max = 25
        quasar_redshifts = self.quasar_rate.redshifts

        inverse_cdf_dict = self.quasar_rate.inverse_cdf_fits_for_redshifts(
            m_min, m_max, quasar_redshifts
        )

        theoretical_samples = []
        for redshift in quasar_redshifts:
            inverse_cdf = inverse_cdf_dict[redshift]
            random_points = np.random.rand(1000)
            theoretical_samples.extend(inverse_cdf(random_points))

        theoretical_samples = np.sort(theoretical_samples)
        theoretical_cdf_values = np.arange(1, len(theoretical_samples) + 1) / len(
            theoretical_samples
        )

        empirical_samples = np.sort(theoretical_samples)
        empirical_cdf_values = np.arange(1, len(empirical_samples) + 1) / len(
            empirical_samples
        )

        _, p_value = ks_2samp(empirical_cdf_values, theoretical_cdf_values)

        # Assert that the p-value is high enough to not reject the null hypothesis
        assert p_value > 0.05, f"KS Test failed: p-value = {p_value}"

    def test_quasar_sample(self):
        # Test data
        m_min = 15
        m_max = 25
        quasar_redshifts = self.quasar_rate.redshifts

        table = self.quasar_rate.quasar_sample(m_min, m_max, quasar_redshifts)
        assert isinstance(
            table, Table
        ), f"Returned object is not an Astropy Table. Type: {type(table)}"


# Running the tests with pytest
if __name__ == "__main__":
    pytest.main()
