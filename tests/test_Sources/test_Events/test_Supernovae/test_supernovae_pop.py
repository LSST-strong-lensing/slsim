from slsim.Sources.Events.Supernovae.supernovae_pop import (
    calculate_star_formation_rate,
    delay_time_distribution,
)
from slsim.Sources.Events.Supernovae.supernovae_pop import SNIaRate
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import IntegrationWarning
import warnings
import numpy as np
import numpy.testing as npt
import pytest


def test_calculate_star_formation_rate():
    z = 0
    npt.assert_almost_equal(calculate_star_formation_rate(z), 0.0118, decimal=4)
    z = 3
    npt.assert_almost_equal(calculate_star_formation_rate(z), 0.1565, decimal=3)


def test_delay_time_distribution():
    t_d = 2
    npt.assert_almost_equal(delay_time_distribution(t_d), 2 ** (-1.08), decimal=4)
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
        # t - t_d has to stay above the age of the universe at z_max, which is
        # the range calculate_SNIa_rate integrates over. Below it the redshift
        # is extrapolated past z_max, where no star formation is modelled and
        # the value is an artifact of the interpolation rather than a physical
        # quantity.
        t_d, t = 1, 5
        npt.assert_almost_equal(
            self.sne_rate._numerator_integrand(t_d, t), 0.1357258, decimal=6
        )

        t_d, t = 0.5, 10
        npt.assert_almost_equal(
            self.sne_rate._numerator_integrand(t_d, t), 0.0861676, decimal=6
        )

    def test_calculate_SNIa_rate(self):
        # (Fig 2 - Oguri and Marshall 2010)
        z_array = [0, 1, 2, 3]
        rate_array = self.sne_rate.calculate_SNIa_rate(z_array)

        npt.assert_approx_equal(rate_array[0], 0.000040895, significant=3)
        npt.assert_approx_equal(rate_array[1], 0.00011900, significant=3)
        npt.assert_approx_equal(rate_array[2], 0.00013488, significant=3)
        npt.assert_approx_equal(rate_array[3], 0.000080099, significant=3)

    def test_calculate_SNIa_rate_does_not_warn(self):
        # The redshift to cosmic time inverse is interpolated with a monotonic
        # spline so that the integrand of the rate is smooth. A piecewise linear
        # interpolation puts a kink at every node and makes the adaptive
        # quadrature exhaust its subintervals.
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            self.sne_rate.calculate_SNIa_rate(np.linspace(0, 5, 100))
        integration_warnings = [
            w for w in raised if issubclass(w.category, IntegrationWarning)
        ]
        assert integration_warnings == []

    def test_calculate_event_rate(self):
        cosmo = FlatLambdaCDM(70, 0.3)
        h = cosmo.H(0).to_value() / 100

        z_array = [0, 1, 2, 3]
        rate_array = self.sne_rate.calculate_SNIa_rate(z_array)
        event_array = self.sne_rate.event_rate(z_array)

        for i in range(len(z_array)):
            npt.assert_approx_equal(rate_array[i] * h, event_array[i], significant=3)


if __name__ == "__main__":
    pytest.main()
