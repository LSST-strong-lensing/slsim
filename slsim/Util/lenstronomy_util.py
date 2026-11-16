from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis

import numpy as np

_NFW_NORM = 1.0 + np.log(0.5)


def _nfw_g(x):
    x = np.maximum(x, 1e-6)
    out = np.empty_like(x, dtype=float)
    m_lt = x < 1.0
    m_eq = x == 1.0
    m_gt = x > 1.0
    if np.any(m_lt):
        xl = x[m_lt]
        out[m_lt] = np.log(xl * 0.5) + np.arccosh(1.0 / xl) / np.sqrt(1.0 - xl**2)
    if np.any(m_eq):
        out[m_eq] = _NFW_NORM
    if np.any(m_gt):
        xg = x[m_gt]
        out[m_gt] = np.log(xg * 0.5) + np.arccos(1.0 / xg) / np.sqrt(xg**2 - 1.0)
    return out


def _hernquist_f(x):
    x = np.maximum(x, 1e-6)
    out = np.empty_like(x, dtype=float)
    m_lt = x < 1.0
    m_eq = x == 1.0
    m_gt = x > 1.0
    if np.any(m_lt):
        xl = x[m_lt]
        s = np.sqrt(1.0 - xl**2)
        out[m_lt] = (1.0 - np.arctanh(s) / s) / (xl**2 - 1.0)
    if np.any(m_eq):
        out[m_eq] = 1.0 / 3.0
    if np.any(m_gt):
        xg = x[m_gt]
        s = np.sqrt(xg**2 - 1.0)
        out[m_gt] = (1.0 - np.arctan(s) / s) / (xg**2 - 1.0)
    return out


def theta_E_spherical_nfw_hernquist(
    rs_nfw=None,
    alpha_rs=None,
    rs_hern=None,
    sigma0=None,
    r_min=1e-2,
    r_max=1e2,
    n_iter=36,
):
    """Solve spherical Einstein radius (alpha(r) = r) for NFW and/or Hernquist.

    Supports scalar or 1D NumPy array inputs. Returns 0 where alpha(r_min) < r_min.
    """
    is_scalar = all(
        v is None or np.ndim(v) == 0 for v in (rs_nfw, alpha_rs, rs_hern, sigma0)
    )
    has_nfw = rs_nfw is not None and alpha_rs is not None
    has_hern = rs_hern is not None and sigma0 is not None
    if not has_nfw and not has_hern:
        raise ValueError("Provide NFW and/or Hernquist parameters.")

    if has_nfw:
        rs_nfw = np.atleast_1d(np.asarray(rs_nfw, dtype=float))
        alpha_rs = np.atleast_1d(np.asarray(alpha_rs, dtype=float))
        n = rs_nfw.shape[0]
    if has_hern:
        rs_hern = np.atleast_1d(np.asarray(rs_hern, dtype=float))
        sigma0 = np.atleast_1d(np.asarray(sigma0, dtype=float))
        n = rs_hern.shape[0]

    def total_alpha(r):
        alpha = np.zeros(n, dtype=float)
        if has_nfw:
            x_n = np.maximum(r, 1e-8) / np.maximum(rs_nfw, 1e-7)
            alpha += (alpha_rs / _NFW_NORM) * _nfw_g(x_n) / x_n
        if has_hern:
            x_h = np.maximum(r, 1e-6) / np.maximum(rs_hern, 1e-7)
            alpha += 2.0 * sigma0 * rs_hern * _hernquist_f(x_h) * x_h
        return alpha

    r_min_arr = np.full(n, r_min, dtype=float)
    valid = total_alpha(r_min_arr) >= r_min
    lo = np.full(n, np.log(r_min), dtype=float)
    hi = np.full(n, np.log(r_max), dtype=float)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r_mid = np.exp(mid)
        gt = total_alpha(r_mid) > r_mid
        lo = np.where(gt, mid, lo)
        hi = np.where(gt, hi, mid)
    theta_e = np.where(valid, np.exp(0.5 * (lo + hi)), 0.0)
    return float(theta_e[0]) if is_scalar else theta_e


def jax_usage(use_jax, lens_mass_model_list):
    """Get all the JAX profiles if enabled from jaxtronomy.

    :param use_jax: if True aims to use all the available jaxtronomy
        profiles
    :type use_jax: bool
    :param lens_mass_model_list:
    :return:
    """

    if use_jax is True:
        _use_jax = []
        for profile in lens_mass_model_list:
            if profile in _JAXXED_MODELS:
                _use_jax.append(True)
            else:
                _use_jax.append(False)
    else:
        _use_jax = False
    return _use_jax


def theta_E_numerical(lens_mass_model_list, kwargs_lens_mass, use_jax=False):
    """Calculate numerically the Einstein radius within a single deflector
    plane.

    :param lens_mass_model_list: lens model list in lenstronomy
        convention
    :param kwargs_lens_mass: list of deflector dictionaries in
        lenstronomy conventions
    :param use_jax: whether JAX-acceleration are used
    :return: Einstein radius
    """
    if (
        len(lens_mass_model_list) in (1, 2)
        and len(set(lens_mass_model_list)) == len(lens_mass_model_list)
        and set(lens_mass_model_list).issubset({"NFW", "HERNQUIST"})
    ):
        from lenstronomy.LensModel.Profiles.nfw import NFW as _LenstronomyNFW
        from lenstronomy.LensModel.Profiles.hernquist import (
            Hernquist as _LenstronomyHernquist,
        )

        r_array = np.logspace(-2, 2, 50)
        kappa_r = np.zeros_like(r_array)
        nfw_prof = _LenstronomyNFW()
        hern_prof = _LenstronomyHernquist()
        for model_name, kw in zip(lens_mass_model_list, kwargs_lens_mass):
            cx = kw.get("center_x", 0.0)
            cy = kw.get("center_y", 0.0)
            if model_name == "NFW":
                rho0 = nfw_prof.alpha2rho0(alpha_Rs=kw["alpha_Rs"], Rs=kw["Rs"])
                kappa_r += nfw_prof.density_2d(
                    r_array + cx, cy, Rs=kw["Rs"], rho0=rho0, center_x=cx, center_y=cy
                )
            elif model_name == "HERNQUIST":
                rho0 = hern_prof.sigma2rho(sigma0=kw["sigma0"], Rs=kw["Rs"])
                kappa_r += hern_prof.density_2d(
                    r_array + cx, cy, rho0=rho0, Rs=kw["Rs"], center_x=cx, center_y=cy
                )
        theta_E_infinity = (
            LensProfileAnalysis.effective_einstein_radius_from_radial_profile(
                r_array, kappa_r
            )
        )
        return np.nan_to_num(theta_E_infinity, nan=0)

    _use_jax = jax_usage(use_jax, lens_mass_model_list)

    lens_model = LensModel(
        lens_model_list=lens_mass_model_list,
        use_jax=_use_jax,
    )

    lens_analysis = LensProfileAnalysis(lens_model=lens_model)

    theta_E_infinity = lens_analysis.effective_einstein_radius(
        kwargs_lens_mass,
        r_min=1e-2,
        r_max=1e2,
        num_points=50,
        spherical_model=True,
    )
    theta_E_infinity = np.nan_to_num(theta_E_infinity, nan=0)
    return theta_E_infinity
