from slsim.Util.cosmo_util import get_cosmo_cache, z_scale_factor, z_time_interp
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
import numpy as np
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


def test_cosmology_distance_cache():
    cache_a = get_cosmo_cache(cosmo)
    cache_b = get_cosmo_cache(cosmo)
    assert cache_a is cache_b

    z_arr = np.array([0.2, 0.8, 2.5])
    npt.assert_allclose(
        cache_a.dd(z_arr), cosmo.angular_diameter_distance(z_arr).value, rtol=1e-4
    )
    npt.assert_allclose(
        cache_a.dc(z_arr), cosmo.comoving_distance(z_arr).value, rtol=1e-4
    )
    npt.assert_allclose(cache_a.efunc(z_arr), cosmo.efunc(z_arr), rtol=1e-4)

    # Check high-redshift auto-expansion above initial z_max=6.0
    npt.assert_allclose(
        cache_a.dd(8.0), cosmo.angular_diameter_distance(8.0).value, rtol=1e-4
    )

    z_l = np.array([0.3, 0.8, 1.5])
    z_s = np.array([1.2, 0.5, 2.0])
    expected_ratio = np.array(
        [
            (
                cosmo.angular_diameter_distance_z1z2(0.3, 1.2)
                / cosmo.angular_diameter_distance(1.2)
            ).value,
            0.0,
            (
                cosmo.angular_diameter_distance_z1z2(1.5, 2.0)
                / cosmo.angular_diameter_distance(2.0)
            ).value,
        ]
    )
    npt.assert_allclose(cache_a.dds_over_ds(z_l, z_s), expected_ratio, rtol=1e-4)

    curved_cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.65)
    curved_cache = get_cosmo_cache(curved_cosmo)
    assert not curved_cache.is_flat
    npt.assert_allclose(
        curved_cache.dds_over_ds(0.4, 1.4),
        (
            curved_cosmo.angular_diameter_distance_z1z2(0.4, 1.4)
            / curved_cosmo.angular_diameter_distance(1.4)
        ).value,
        rtol=1e-5,
    )
