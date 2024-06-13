from slsim.Sources.Supernovae.supernovae_pop import (
    calculate_star_formation_rate,
    delay_time_distribution,
)
from slsim.Sources.Supernovae.supernovae_pop import SNIaRate
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import numpy.testing as npt


def test_calculate_star_formation_rate():
    z = 0
    npt.assert_almost_equal(calculate_star_formation_rate(z), 0.0118, decimal=4)
    z = 3
    npt.assert_almost_equal(calculate_star_formation_rate(z), 0.1565, decimal=3)


def test_delay_time_distribution():
    t_d = -1
    npt.assert_equal(delay_time_distribution(t_d), np.nan)
    t_d = 5
    npt.assert_almost_equal(delay_time_distribution(t_d), 5 ** (-1.08), decimal=4)


class TestSNIaRate:
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.z_max = 10
        self.sne_rate = SNIaRate(
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

    def test_z_from_time(self):
        z_true = 0
        t = self.cosmo.age(z_true)
        z_est = self.sne_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

        z_true = 4
        t = self.cosmo.age(z_true)
        z_est = self.sne_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

        z_true = 7
        t = self.cosmo.age(z_true)
        z_est = self.sne_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

        z_true = 10
        t = self.cosmo.age(z_true)
        z_est = self.sne_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

    def test_numerator_integrand(self):
        t_d, t = -1, 1
        npt.assert_equal(self.sne_rate._numerator_integrand(t_d, t), np.nan)
        t_d, t = 1, -1
        npt.assert_equal(self.sne_rate._numerator_integrand(t_d, t), np.nan)
        t_d, t = 1, 1
        npt.assert_almost_equal(
            self.sne_rate._numerator_integrand(t_d, t), 0.0002559, decimal=4
        )

    def test_calculate_SNIa_rate(self):
        z = -1
        npt.assert_equal(self.sne_rate.calculate_SNIa_rate(z), np.nan)
        # (Fig 2 - Oguri and Marshall 2010)
        z = 0
        npt.assert_almost_equal(
            self.sne_rate.calculate_SNIa_rate(z), 0.000041006, decimal=3
        )
        z = 1
        npt.assert_almost_equal(
            self.sne_rate.calculate_SNIa_rate(z), 0.0001191, decimal=3
        )
        z = 2
        npt.assert_almost_equal(
            self.sne_rate.calculate_SNIa_rate(z), 0.0001349, decimal=3
        )
        z = 3
        npt.assert_almost_equal(
            self.sne_rate.calculate_SNIa_rate(z), 0.00008008, decimal=3
        )
