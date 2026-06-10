from slsim.Util.cosmo_util import z_scale_factor
from slsim.Util.cosmo_util import z_time_interp
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def test_z_scale_factor():
    z_old = 1.0
    z_new = 1.0
    scale_factor = z_scale_factor(z_old, z_new, cosmo)
    npt.assert_almost_equal(scale_factor.value, 1.0, decimal=3)

    z_old = 0.5
    z_new = 1.0
    scale_factor = z_scale_factor(z_old, z_new, cosmo)
    npt.assert_almost_equal(scale_factor.value, 0.762, decimal=3)


def test_z_time_interp():
    z_max = 10
    z_from_time = z_time_interp(cosmo, z_max)

    z_true = 0
    t = cosmo.age(z_true).to_value()
    z_est = z_from_time(t)
    npt.assert_almost_equal(z_est, z_true, decimal=3)

    z_true = 4
    t = cosmo.age(z_true).to_value()
    z_est = z_from_time(t)
    npt.assert_almost_equal(z_est, z_true, decimal=3)

    z_true = 7
    t = cosmo.age(z_true).to_value()
    z_est = z_from_time(t)
    npt.assert_almost_equal(z_est, z_true, decimal=3)  #

    z_true = 10
    t = cosmo.age(z_true).to_value()
    z_est = z_from_time(t)
    npt.assert_almost_equal(z_est, z_true, decimal=3)
