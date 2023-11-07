from slsim.ParamDistributions.gaussian_mixture_model import GaussianMixtureModel
from slsim.Deflectors.velocity_dispersion import vel_disp_composite_model
from astropy.cosmology import FlatLambdaCDM
from pydantic import ValidationError
import pytest
import random

"""
Test for the parameter checking rountines in slsim. We want to make sure we're using
functions/classes that are actually in slsim, rather than ones in the /tests folder. 
This is because the parameter checking routine discovers defaults by importing from 
a standard location inside the slsim package. If we import from the /tests folder, 
this will fail.
"""


def test_all_kwargs_init(good_inputs: dict):
    gmm = GaussianMixtureModel(**good_inputs)
    assert (
        gmm.means == good_inputs["means"]
        and gmm.stds == good_inputs["stds"]
        and gmm.weights == good_inputs["weights"]
    )


def test_all_args_init(good_inputs: dict):
    inputs = list(good_inputs.values())
    gmm = GaussianMixtureModel(*inputs)
    assert gmm.means == good_inputs["means"]


def test_all_args_method(good_model: GaussianMixtureModel):
    output = good_model.rvs(100)
    assert len(output) == 100


def test_all_kwargs_method(good_model: GaussianMixtureModel):
    output = good_model.rvs(size=100)
    assert len(output) == 100


def test_all_args_function(vel_disp_inputs: dict):
    inputs = list(vel_disp_inputs.values())
    output = vel_disp_composite_model(*inputs)
    assert isinstance(output, float)


def test_all_kwargs_function(vel_disp_inputs: dict):
    output = vel_disp_composite_model(**vel_disp_inputs)
    assert isinstance(output, float)


def test_mixture_init(good_inputs: dict):
    good_inputs_keys = list(good_inputs.keys())
    input_args = [good_inputs[k] for k in good_inputs_keys[:2]]
    input_kwargs = {k: good_inputs[k] for k in good_inputs_keys[2:]}

    gmm = GaussianMixtureModel(*input_args, **input_kwargs)
    assert (
        gmm.means == good_inputs["means"]
        and gmm.stds == good_inputs["stds"]
        and gmm.weights == good_inputs["weights"]
    )


def test_mixture_function(vel_disp_inputs: dict):
    vel_disp_inputs_keys = list(vel_disp_inputs.keys())
    input_args = [vel_disp_inputs[k] for k in vel_disp_inputs_keys[:3]]
    input_kwargs = {k: vel_disp_inputs[k] for k in vel_disp_inputs_keys[3:]}

    output = vel_disp_composite_model(*input_args, **input_kwargs)
    assert isinstance(output, float)


def test_shuffle_init(good_inputs: dict):
    keys = list(good_inputs.keys())
    random.shuffle(keys)
    input = {k: good_inputs[k] for k in keys}
    gmm = GaussianMixtureModel(**input)
    assert (
        gmm.means == good_inputs["means"]
        and gmm.stds == good_inputs["stds"]
        and gmm.weights == good_inputs["weights"]
    )


def test_shuffle_function(vel_disp_inputs: dict):
    keys = list(vel_disp_inputs.keys())
    random.shuffle(keys)
    input = {k: vel_disp_inputs[k] for k in keys}
    output = vel_disp_composite_model(**input)
    assert isinstance(output, float)


def test_failure_init(good_inputs: dict):
    good_inputs_keys = list(good_inputs.keys())
    input_args = [good_inputs[k] for k in good_inputs_keys[:2]]
    input_kwargs = {k: good_inputs[k] for k in good_inputs_keys[2:]}
    input_kwargs["weights"] = [0.2, 0.3, 0.4]
    with pytest.raises(ValueError):
        _ = GaussianMixtureModel(*input_args, **input_kwargs)


def test_failure_method(good_model: GaussianMixtureModel):
    with pytest.raises(ValidationError):
        _ = good_model.rvs(-50)


def test_failure_function(vel_disp_inputs: dict):
    input_kwargs = vel_disp_inputs
    input_kwargs["m_star"] = -50
    with pytest.raises(ValidationError):
        _ = vel_disp_composite_model(**input_kwargs)


@pytest.fixture
def cosmology():
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def vel_disp_inputs(cosmology):
    return {
        "r": 5,
        "m_star": 10**10,
        "rs_star": 30,
        "m_halo": 10**14,
        "c_halo": 3,
        "cosmo": cosmology,
    }


@pytest.fixture
def good_inputs():
    return {"means": [1, 2, 3], "stds": [1, 2, 3], "weights": [0.2, 0.3, 0.5]}


@pytest.fixture
def good_model(good_inputs: dict):
    return GaussianMixtureModel(**good_inputs)
