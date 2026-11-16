from slsim.Util.cosmo_util import (
    _MAX_FALLBACK_COSMOS,
    get_cosmo_cache,
    z_scale_factor,
    z_time_interp,
)
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
    assert curved_cache.dds_over_ds(1.4, 0.4) == 0.0
    npt.assert_allclose(
        curved_cache.dds_over_ds(np.array([0.4, 1.4]), np.array([1.4, 0.4])),
        np.array(
            [
                (
                    curved_cosmo.angular_diameter_distance_z1z2(0.4, 1.4)
                    / curved_cosmo.angular_diameter_distance(1.4)
                ).value,
                0.0,
            ]
        ),
        rtol=1e-5,
    )

    # Scalar flat dds_over_ds, negative-redshift exact fallback, and z=100 helper
    assert cache_a.dds_over_ds(0.5, 0.3) == 0.0
    npt.assert_allclose(cache_a.dds_over_ds(0.3, 1.2), expected_ratio[0], rtol=1e-4)
    npt.assert_allclose(cache_a.efunc(-0.05), cosmo.efunc(-0.05), rtol=1e-6)
    npt.assert_allclose(
        cache_a.efunc(np.array([-0.05, 0.5])),
        cosmo.efunc(np.array([-0.05, 0.5])),
        rtol=1e-4,
    )
    dd_flat, ds_flat, dds_flat = cache_a.dd_ds_dds_at_z100(np.array([0.3, 0.6]))
    npt.assert_allclose(
        ds_flat, cosmo.angular_diameter_distance(100.0).value, rtol=1e-6
    )
    npt.assert_allclose(
        dd_flat, cosmo.angular_diameter_distance([0.3, 0.6]).value, rtol=1e-4
    )
    npt.assert_allclose(
        dds_flat,
        cosmo.angular_diameter_distance_z1z2([0.3, 0.6], 100.0).value,
        rtol=1e-4,
    )
    dd_c, ds_c, dds_c = curved_cache.dd_ds_dds_at_z100(np.array([0.3, 0.6]))
    npt.assert_allclose(
        dds_c,
        curved_cosmo.angular_diameter_distance_z1z2([0.3, 0.6], 100.0).value,
        rtol=1e-6,
    )

    # Weakref-capable cosmology object and bounded fallback cache eviction
    class _WeakrefCosmoWrapper:
        def __init__(self, base):
            self._base = base
            self.Ok0 = base.Ok0

        def angular_diameter_distance(self, z):
            return self._base.angular_diameter_distance(z)

        def comoving_distance(self, z):
            return self._base.comoving_distance(z)

        def efunc(self, z):
            return self._base.efunc(z)

    wrapped = _WeakrefCosmoWrapper(cosmo)
    w_cache_1 = get_cosmo_cache(wrapped)
    w_cache_2 = get_cosmo_cache(wrapped)
    assert w_cache_1 is w_cache_2

    for i in range(_MAX_FALLBACK_COSMOS + 2):
        get_cosmo_cache(FlatLambdaCDM(H0=65.0 + i, Om0=0.3))
