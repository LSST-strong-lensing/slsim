from slsim.Sources.Events.BNSMerger.BNSMerger_pop import BNSMergerRate
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt
import numpy as np
import pytest

class TestBNSMergerRate:
    def setup_method(self):
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.z_max = 10
        self.bnsm_rate = BNSMergerRate(
            cosmo=self.cosmo,
            z_max=self.z_max,
        )
        
    def test_calculate_star_formation_rate(self):
        z = 0
        npt.assert_almost_equal(self.bnsm_rate.calculate_star_formation_rate(z), 0.01505, decimal=3)
        z = 3
        npt.assert_almost_equal(self.bnsm_rate.calculate_star_formation_rate(z), 0.10854, decimal=3)

    def test_delay_time_distribution(self):
        t_d = 2
        npt.assert_almost_equal(self.bnsm_rate.delay_time_distribution(t_d), 1 / (2 * (np.log(13.8/0.020))), decimal=4)
        t_d = 5
        npt.assert_almost_equal(self.bnsm_rate.delay_time_distribution(t_d), 1 / (5 * (np.log(13.8/0.020))), decimal=4)

    def test_z_from_time(self):
        z_true = 0
        t = self.cosmo.age(z_true)
        z_est = self.bnsm_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

        z_true = 4
        t = self.cosmo.age(z_true)
        z_est = self.bnsm_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

        z_true = 7
        t = self.cosmo.age(z_true)
        z_est = self.bnsm_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)

        z_true = 10
        t = self.cosmo.age(z_true)
        z_est = self.bnsm_rate.z_from_time(t)
        npt.assert_almost_equal(z_est, z_true, decimal=3)
    
    def test_numerator_integrand(self):
        t_d, t = 1, 1
        npt.assert_almost_equal(
            self.bnsm_rate._numerator_integrand(t_d, t), 0.00013105, decimal=5
        )
    
    def test_calculate_event_rate(self):
        z_array = [0, 1, 2, 3]
        rate_array = self.bnsm_rate.calculate_event_rate(z_array)
    
        npt.assert_almost_equal(rate_array[0], 0.03081, decimal=3)
        npt.assert_almost_equal(rate_array[1], 0.09757, decimal=3)
        npt.assert_almost_equal(rate_array[2], 0.09297, decimal=3)
        npt.assert_almost_equal(rate_array[3], 0.05915, decimal=3)

if __name__ == "__main__":
    pytest.main()