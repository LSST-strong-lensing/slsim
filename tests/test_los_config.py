from slsim.ParamDistributions.los_config import LOSConfig


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
