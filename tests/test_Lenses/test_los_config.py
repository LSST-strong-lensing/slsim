from slsim.LOS.los_pop import LOSPop
import os
from astropy.cosmology import FlatLambdaCDM
import pytest

path = os.path.dirname(__file__)
module_path, _ = os.path.split(path)
mother_path = os.path.dirname(os.path.dirname(path))

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

path_to_h5 = os.path.join(mother_path, "data/glass/no_nonlinear_distributions.h5")


def test_default_settings():
    config = LOSPop()
    assert config.mixgauss_gamma is False
    assert config.mixgauss_means is None
    assert config.mixgauss_stds is None
    assert config.mixgauss_weights is None
    assert config.los_bool is True
    assert config.nonlinear_los_bool is False
    assert config.nonlinear_correction_path is None
    assert config.no_correction_path is None


@pytest.fixture
def los_pop():
    # Create an instance of LOSConfig with some default settings
    los_pop = LOSPop(
        los_bool=True,
        mixgauss_gamma=False,
        mixgauss_means=[0.1],
        mixgauss_stds=[0.01],
        mixgauss_weights=[1],
        nonlinear_los_bool=False,
        nonlinear_correction_path=None,
        no_correction_path=None,
    )
    return los_pop


def test_no_los_effects(los_pop):
    los_pop.los_bool = False
    los_pop.nonlinear_los_bool = False
    source = 0.5
    deflector = 0.2
    los_class = los_pop.draw_los(source, deflector)
    kappa = los_class.convergence
    gamma1, gamma2 = los_class.shear
    assert (kappa, gamma1, gamma2) == (0, 0, 0)


def test_gaussian_mixture_model_gamma(los_pop):
    los_pop.mixgauss_gamma = True
    source = 0.4
    deflector = 0.2
    los_class = los_pop.draw_los(source, deflector)
    kappa = los_class.convergence
    gamma1, gamma2 = los_class.shear
    assert isinstance(gamma1, float)
    assert isinstance(gamma2, float)
    assert isinstance(kappa, float)


def test_conflicting_settings_error(los_pop):
    los_pop.mixgauss_gamma = True
    los_pop.nonlinear_los_bool = True
    source = 0.3
    deflector = 0.1
    with pytest.raises(ValueError):
        los_pop.draw_los(source, deflector)


def test_nonlinear_los_corrections(los_pop):
    los_pop.nonlinear_los_bool = False
    los_pop.no_correction_path = path_to_h5
    source = 0.2
    deflector = 0.1
    los_class = los_pop.draw_los(source, deflector)
    kappa = los_class.convergence
    gamma1, gamma2 = los_class.shear
    assert isinstance(gamma1, float)
    assert isinstance(gamma2, float)
    assert isinstance(kappa, float)
