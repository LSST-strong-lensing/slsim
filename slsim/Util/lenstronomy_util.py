from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis

import numpy as np


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
