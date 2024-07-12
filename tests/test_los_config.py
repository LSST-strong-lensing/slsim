from slsim.ParamDistributions.los_config import LOSConfig
import os
from astropy.cosmology import FlatLambdaCDM
import pytest


path = os.path.dirname(__file__)
module_path, _ = os.path.split(path)
mother_path = os.path.dirname(path)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

path_to_h5 = os.path.join(mother_path, "data/glass/no_nonlinear_distributions.h5")


def test_default_settings():
    config = LOSConfig()
    assert config.mixgauss_gamma is False
    assert config.mixgauss_means is None
    assert config.mixgauss_stds is None
    assert config.mixgauss_weights is None
    assert config.los_bool is True
    assert config.nonlinear_los_bool is False
    assert config.nonlinear_correction_path is None
    assert config.no_correction_path is None


@pytest.fixture
def los_config():
    # Create an instance of LOSConfig with some default settings
    config = LOSConfig(
        los_bool=True,
        mixgauss_gamma=False,
        mixgauss_means=[0.1],
        mixgauss_stds=[0.01],
        mixgauss_weights=[1],
        nonlinear_los_bool=False,
        nonlinear_correction_path=None,
        no_correction_path=None,
    )
    return config


def test_no_los_effects(los_config):
    los_config.los_bool = False
    los_config.nonlinear_los_bool = False
    source = 0.5
    deflector = 0.2
    assert los_config.calculate_los_linear_distortions(source, deflector) == (0, 0, 0)


def test_gaussian_mixture_model_gamma(los_config):
    los_config.mixgauss_gamma = True
    source = 0.4
    deflector = 0.2
    gamma1, gamma2, kappa = los_config.calculate_los_linear_distortions(
        source, deflector
    )
    assert isinstance(gamma1, float)
    assert isinstance(gamma2, float)
    assert isinstance(kappa, float)


def test_conflicting_settings_error(los_config):
    los_config.mixgauss_gamma = True
    los_config.nonlinear_los_bool = True
    source = 0.3
    deflector = 0.1
    with pytest.raises(ValueError):
        los_config.calculate_los_linear_distortions(source, deflector)


def test_nonlinear_los_corrections(los_config):
    los_config.nonlinear_los_bool = False
    los_config.no_correction_path = path_to_h5
    source = 0.2
    deflector = 0.1
    g1, g2, kappa = los_config.calculate_los_linear_distortions(source, deflector)
    assert isinstance(g1, float)
    assert isinstance(g2, float)
    assert isinstance(kappa, float)
