import numpy as np
from slsim.Sources.Events.BNSMerger.kilonova import Kilonova
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import pytest


@pytest.fixture
def kilonova_parameters():
    return {
        "mej_1": 0.01,
        "mej_2": 0.02,
        "mej_3": 0.03,
        "vej_1": 0.1,
        "vej_2": 0.2,
        "vej_3": 0.3,
        "kappa_1": 0.5,
        "kappa_2": 3.0,
        "kappa_3": 10.0,
        "temperature_floor_1": 5000,
        "temperature_floor_2": 4000,
        "temperature_floor_3": 3000,
        "kappa_gamma": 10,
    }


@pytest.fixture
def kilonova_class(kilonova_parameters):
    KN = Kilonova(
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        redshift=0.1,
        model_name="mosfit_kilonova",
        mag_zpsys="AB",
        dense_resolution=50,
        **kilonova_parameters,
    )

    return KN


def test_kilonova_mag(kilonova_class):
    time = np.array([0.5, 1.0, 2.0])
    light_curve = kilonova_class.get_apparent_magnitude(time=time, band="r")

    assert isinstance(light_curve, Table)
    assert "time" in light_curve.colnames
    assert "magnitude" in light_curve.colnames
    assert len(light_curve) > 0

    magnitudes = light_curve["magnitude"]
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]

    assert np.any(np.isfinite(magnitudes))
    assert np.all(np.isfinite(magnitudes) | np.isposinf(magnitudes))
    assert np.all(finite_magnitudes > 0)


def test_kilonova_missing_parameters(kilonova_parameters):
    # Test that omitting a required model parameter raises an error.
    incomplete_parameters = kilonova_parameters.copy()
    incomplete_parameters.pop("mej_1")
    kilonova = Kilonova(
        redshift=0.1,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        **incomplete_parameters,
    )

    with pytest.raises(TypeError, match="mej_1"):
        kilonova.get_apparent_magnitude(time=np.array([1.0]), band="r")


def test_kilonova_invalid_model_name(kilonova_parameters):
    # Test that an unavailable Redback model name raises an error.
    with pytest.raises(ValueError):
        Kilonova(
            redshift=0.1,
            model_name="not_a_kilonova_model",
            **kilonova_parameters,
        )


def test_kilonova_external_modeldir_not_supported(kilonova_parameters):
    with pytest.raises(NotImplementedError):
        Kilonova(
            redshift=0.1,
            modeldir="some/path",
            **kilonova_parameters,
        )


def test_kilonova_invalid_redshift():
    with pytest.raises(ValueError, match="redshift must be finite and positive"):
        Kilonova(redshift=0.0)


def test_kilonova_model_parameters_dict():
    with pytest.raises(TypeError, match="model_parameters must be a dictionary"):
        Kilonova(redshift=0.1, model_parameters=[])


if __name__ == "__main__":
    pytest.main()
