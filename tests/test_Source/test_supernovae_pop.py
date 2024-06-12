from slsim.Sources.Supernovae.supernovae_pop import (
    calculate_star_formation_rate,
    t_d_power,
)
from slsim.Sources.Supernovae.supernovae_pop import SNIaRate
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import numpy.testing as npt

cosmo = FlatLambdaCDM(70, 0.3)
sne_rate = SNIaRate(cosmo=cosmo)


def test_calculate_star_formation_rate():
    z = -1
    npt.assert_equal(calculate_star_formation_rate(z), np.nan)
    z = 0
    npt.assert_almost_equal(calculate_star_formation_rate(z), 0.0118, decimal=4)
    z = 3
    npt.assert_almost_equal(calculate_star_formation_rate(z), 0.1565, decimal=3)


def test_t_d_power():
    t_d = -1
    npt.assert_equal(t_d_power(t_d), np.nan)
    t_d = 5
    npt.assert_almost_equal(t_d_power(t_d), 0.1758, decimal=4)


def test_z_from_time():
    z_true = 1
    t = cosmo.age(z_true)
    z_est = sne_rate.z_from_time(t)
    npt.assert_almost_equal(z_est, z_true, decimal=3)


def test_numerator_integrand():
    t_d, t = -1, 1
    npt.assert_equal(sne_rate._numerator_integrand(t_d, t), np.nan)
    t_d, t = 1, -1
    npt.assert_equal(sne_rate._numerator_integrand(t_d, t), np.nan)
    t_d, t = 1, 1
    npt.assert_almost_equal(
        sne_rate._numerator_integrand(t_d, t), 8.9363e-12, decimal=4
    )


def test_calculate_SNIa_rate():
    z = -1
    npt.assert_equal(sne_rate.calculate_SNIa_rate(z), np.nan)
    z = 1.3
    npt.assert_almost_equal(sne_rate.calculate_SNIa_rate(z), 6.661537e-5, decimal=2)
