import numpy as np
import numpy.testing as npt
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.lens_model import LensModel

from slsim.Util import lenstronomy_util


def test_jax_usage():
    _use_jax = lenstronomy_util.jax_usage(use_jax=False, lens_mass_model_list=["SHEAR"])
    assert _use_jax is False

    _use_jax = lenstronomy_util.jax_usage(use_jax=True, lens_mass_model_list=["SHEAR"])
    assert len(_use_jax) == 1
    assert _use_jax[0] is True


def test_theta_e_spherical_nfw_hernquist_and_numerical():
    kwargs_nfw = {"alpha_Rs": 1.2, "Rs": 15.0, "center_x": 0.0, "center_y": 0.0}
    kwargs_hern = {"sigma0": 2.5, "Rs": 0.8, "center_x": 0.0, "center_y": 0.0}

    fast_val = lenstronomy_util.theta_E_spherical_nfw_hernquist(
        rs_nfw=kwargs_nfw["Rs"],
        alpha_rs=kwargs_nfw["alpha_Rs"],
        rs_hern=kwargs_hern["Rs"],
        sigma0=kwargs_hern["sigma0"],
    )
    num_val = lenstronomy_util.theta_E_numerical(
        ["NFW", "HERNQUIST"], [kwargs_nfw, kwargs_hern]
    )
    lens_analysis = LensProfileAnalysis(LensModel(lens_model_list=["NFW", "HERNQUIST"]))
    ref_val = lens_analysis.effective_einstein_radius(
        [kwargs_nfw, kwargs_hern],
        r_min=1e-2,
        r_max=1e2,
        num_points=50,
        spherical_model=True,
    )
    npt.assert_allclose(num_val, ref_val, rtol=1e-5)
    npt.assert_allclose(fast_val, ref_val, rtol=1e-2)

    arr_vals = lenstronomy_util.theta_E_spherical_nfw_hernquist(
        rs_nfw=np.array([5.0, 10.0]),
        alpha_rs=np.array([2.5, 0.001]),
    )
    assert arr_vals.shape == (2,)
    assert arr_vals[0] > 0.1
    assert arr_vals[1] == 0.0
