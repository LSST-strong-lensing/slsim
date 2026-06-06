from slsim.Sources.Events.BNSMerger.bns_merger_pop import BNSMergerRate
from slsim.Sources.Events.BNSMerger.bns_merger_pop import norm_delay_time_distribution
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
import numpy as np
import pytest


def test_norm_delay_time_distribution():
    t_d_min = 0.020
    t_d_max = 13.8

    t_d = 2
    npt.assert_almost_equal(
        norm_delay_time_distribution(t_d, t_d_min, t_d_max),
        1 / (2 * (np.log(13.8 / 0.020))),
        decimal=4,
    )
    t_d = 5
    npt.assert_almost_equal(
        norm_delay_time_distribution(t_d, t_d_min, t_d_max),
        1 / (5 * (np.log(13.8 / 0.020))),
        decimal=4,
    )


class TestBNSMergerRate:
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.z_max = 10
        self.bnsm_rate = BNSMergerRate(
            cosmo=self.cosmo,
            z_max=self.z_max,
        )

    def test_binary_formation_rate(self):
        z = 0
        npt.assert_almost_equal(
            self.bnsm_rate.binary_formation_rate(z), 0.01505, decimal=3
        )
        z = 3
        npt.assert_almost_equal(
            self.bnsm_rate.binary_formation_rate(z), 0.10854, decimal=3
        )

    def test_numerator_integrand(self):
        t_d, t = 1, 1
        npt.assert_almost_equal(
            self.bnsm_rate._numerator_integrand(t_d, t), 0.00013105, decimal=5
        )

    def test_event_rate(self):
        z_array = [0, 1, 2, 3]
        rate_array = self.bnsm_rate.event_rate(z_array)

        npt.assert_almost_equal(rate_array[0] * 1e9, 320.00, decimal=3)
        npt.assert_almost_equal(rate_array[1] * 1e9, 1014.585, decimal=3)
        npt.assert_almost_equal(rate_array[2] * 1e9, 966.806, decimal=3)
        npt.assert_almost_equal(rate_array[3] * 1e9, 614.176, decimal=3)


if __name__ == "__main__":
    pytest.main()
