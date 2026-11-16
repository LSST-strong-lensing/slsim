import weakref
import numpy as np
import scipy.interpolate as interp

_COSMO_CACHE = weakref.WeakKeyDictionary()
_COSMO_FALLBACK_CACHE = []
_MAX_FALLBACK_COSMOS = 16


class CosmologyDistanceCache:
    """Fast interpolated angular diameter, comoving distance, and E(z) lookup."""

    def __init__(self, cosmo, z_max=6.0, num_points=2001):
        self.cosmo = cosmo
        self.z_max = float(z_max)
        self.num_points = int(num_points)
        self.is_flat = abs(float(getattr(cosmo, "Ok0", 0.0))) < 1e-12
        self._build_grid(self.z_max)
        self._z100_cache = None

    def _build_grid(self, z_max):
        self.z_max = float(z_max)
        n_pts = max(self.num_points, int(np.ceil(self.z_max / 0.003)) + 1)
        self.z_grid = np.linspace(0.0, self.z_max, n_pts)
        self.dd_grid = self.cosmo.angular_diameter_distance(self.z_grid).value
        self.dc_grid = self.cosmo.comoving_distance(self.z_grid).value
        self.ef_grid = np.asarray(self.cosmo.efunc(self.z_grid), dtype=float)

    def _ensure_z_max(self, z):
        max_z = float(np.max(z))
        if max_z > self.z_max:
            self._build_grid(max(max_z * 1.1, self.z_max * 2.0))

    def _interp(self, z, grid_attr, exact_fn):
        z_arr = np.asarray(z, dtype=float)
        if np.ndim(z_arr) == 0:
            zv = float(z_arr)
            if zv < 0.0:
                return float(exact_fn(zv))
            self._ensure_z_max(zv)
            return float(np.interp(zv, self.z_grid, getattr(self, grid_attr)))
        if np.any(z_arr < 0.0):
            return np.asarray(exact_fn(z_arr), dtype=float)
        self._ensure_z_max(z_arr)
        return np.interp(z_arr, self.z_grid, getattr(self, grid_attr))

    def dd(self, z):
        """Return angular diameter distance [Mpc] at redshift z."""
        return self._interp(
            z, "dd_grid", lambda zv: self.cosmo.angular_diameter_distance(zv).value
        )

    def dc(self, z):
        """Return comoving distance [Mpc] at redshift z."""
        return self._interp(
            z, "dc_grid", lambda zv: self.cosmo.comoving_distance(zv).value
        )

    def efunc(self, z):
        """Return dimensionless Hubble parameter E(z)."""
        return self._interp(z, "ef_grid", self.cosmo.efunc)

    def dds_over_ds(self, z_lens, z_source):
        """Return D_ds / D_s distance ratio."""
        z_l = np.asarray(z_lens, dtype=float)
        z_s = np.asarray(z_source, dtype=float)
        if self.is_flat:
            dc_l = self.dc(z_l)
            dc_s = self.dc(z_s)
            if np.ndim(dc_l) == 0 and np.ndim(dc_s) == 0:
                if z_s <= z_l or z_s <= 0.0:
                    return 0.0
                return float((dc_s - dc_l) / dc_s)
            ratio = np.zeros(
                np.broadcast_shapes(np.shape(z_l), np.shape(z_s)), dtype=float
            )
            valid = (z_s > z_l) & (z_s > 0.0)
            if np.any(valid):
                dc_s_b = np.broadcast_to(dc_s, ratio.shape)
                dc_l_b = np.broadcast_to(dc_l, ratio.shape)
                ratio[valid] = (dc_s_b[valid] - dc_l_b[valid]) / dc_s_b[valid]
            return ratio
        ds = self.cosmo.angular_diameter_distance(z_s).value
        dds = self.cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
        if np.ndim(z_l) == 0 and np.ndim(z_s) == 0:
            return float(dds / ds) if z_s > z_l else 0.0
        return np.where(z_s > z_l, dds / ds, 0.0)

    def dd_ds_dds_at_z100(self, z_lens):
        """Return (D_d, D_s(100), D_ds(z_lens, 100)) [Mpc]."""
        if self._z100_cache is None:
            ds_100 = float(self.cosmo.angular_diameter_distance(100.0).value)
            dc_100 = float(self.cosmo.comoving_distance(100.0).value)
            self._z100_cache = (ds_100, dc_100)
        ds_100, dc_100 = self._z100_cache
        dd_l = self.dd(z_lens)
        if self.is_flat:
            dc_l = self.dc(z_lens)
            dds = (dc_100 - dc_l) / 101.0
        else:
            dds = np.asarray(
                self.cosmo.angular_diameter_distance_z1z2(z_lens, 100.0).value,
                dtype=float,
            )
        return dd_l, ds_100, dds


def get_cosmo_cache(cosmo):
    """Return a cached CosmologyDistanceCache for an astropy cosmology."""
    try:
        cache = _COSMO_CACHE.get(cosmo)
        if cache is None:
            cache = CosmologyDistanceCache(cosmo)
            _COSMO_CACHE[cosmo] = cache
        return cache
    except TypeError:
        for cached_cosmo, cache in _COSMO_FALLBACK_CACHE:
            if cached_cosmo is cosmo:
                return cache
        cache = CosmologyDistanceCache(cosmo)
        if len(_COSMO_FALLBACK_CACHE) >= _MAX_FALLBACK_COSMOS:
            _COSMO_FALLBACK_CACHE.pop(0)
        _COSMO_FALLBACK_CACHE.append((cosmo, cache))
        return cache


def z_scale_factor(z_old, z_new, cosmo):
    """
    :param z_old: The original redshift.
    :type z_old: float

    :param z_new: The redshift where the object will be placed.
    :type z_new: float

    :param cosmo: The cosmology object. Defaults to a FlatLambdaCDM model if None.
    :type cosmo: astropy.cosmology.FLRW, optional

    :return: The multiplicative pixel size scaling factor.
    :rtype: float
    """
    # Calculate angular diameter distance scaling factor
    return cosmo.angular_diameter_distance(z_old) / cosmo.angular_diameter_distance(
        z_new
    )


def z_time_interp(cosmo, z_max):
    """Calculates redshift given cosmic time.

    :param cosmo: cosmology used to calculate cosmic time
    :type cosmo: astropy.cosmology object
    :param z_max: maximum redshift for interpolation
    :type z_max: float
    :return: interpolation function that returns redshift for a given
        cosmic time
    :return type: scipy.interpolate.interp1d
    """
    z_array = np.linspace(0, z_max, 1000)
    z_array = z_array[::-1]
    t_array = cosmo.age(z_array).to_value()

    return interp.interp1d(t_array, z_array, fill_value="extrapolate")
