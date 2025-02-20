from slsim.Sources.Supernovae.supernovae_lightcone import SNeLightcone
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from scipy.stats import ks_2samp
import numpy as np
from astropy import units
import numpy.testing as npt
import pytest


class TestSNeLightcone:
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.redshifts = np.linspace(0, 5, 20)
        self.sky_area = Quantity(value=0.05, unit="deg2")
        self.noise = False
        self.time_interval = 1 * units.year

        self.sne_lightcone = SNeLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
        )

    def test_convert_density(self):
        npt.assert_(isinstance(self.time_interval, units.Quantity))

        self.time_interval = 2 * units.day
        self.sne_lightcone = SNeLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
        )
        test_density = self.sne_lightcone.density[2]
        npt.assert_almost_equal(
            test_density,
            (((0.0001266925 / 365.25) * 2) / (1 + 0.52631579) ** 3),
            decimal=4,
        )

        self.time_interval = 1 * units.year
        self.sne_lightcone = SNeLightcone(
            cosmo=self.cosmo,
            redshifts=self.redshifts,
            sky_area=self.sky_area,
            noise=self.noise,
            time_interval=self.time_interval,
        )
        test_density = self.sne_lightcone.density[0]
        npt.assert_almost_equal(test_density, 5.87137528e-05, decimal=4)

    def test_supernovae_sample(self):
        # Observed SN Ia redshift locations using return_supernovae_sample()
        sample_output = self.sne_lightcone.supernovae_sample()

        # Observed counts using Lightcone()
        observed_counts, bin_edges = np.histogram(
            sample_output, bins=self.redshifts, density=True
        )

        # Expected counts based SN Ia comoving density
        dN_dz = (
            self.cosmo.differential_comoving_volume(self.redshifts) * self.sky_area
        ).to_value("Mpc3")
        dN_dz *= self.sne_lightcone.density
        bin_widths = np.diff(self.redshifts)
        expected_counts = dN_dz[:-1] * bin_widths

        # Normalize expected counts
        expected_counts *= len(sample_output) / np.sum(expected_counts)

        # KS test to compare observed and expected distributions
        observed_cdf = np.cumsum(observed_counts)
        observed_cdf /= observed_cdf[-1]

        expected_cdf = np.cumsum(expected_counts)
        expected_cdf /= expected_cdf[-1]

        _, p_value = ks_2samp(observed_cdf, expected_cdf)

        assert p_value > 0.05, f"KS Test failed: p-value = {p_value}"


if __name__ == "__main__":
    pytest.main()
