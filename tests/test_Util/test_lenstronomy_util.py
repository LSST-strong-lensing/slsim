from slsim.Util import lenstronomy_util


def test_jax_usage():
    _use_jax = lenstronomy_util.jax_usage(use_jax=False, lens_mass_model_list=["SHEAR"])
    assert _use_jax is False

    _use_jax = lenstronomy_util.jax_usage(use_jax=True, lens_mass_model_list=["SHEAR"])
    assert len(_use_jax) == 1
    assert _use_jax[0] is True
