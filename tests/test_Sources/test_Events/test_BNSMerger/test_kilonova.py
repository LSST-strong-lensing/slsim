import numpy as np
from slsim.Sources.Events.BNSMerger.kilonova import Kilonova
import numpy.testing as npt
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
        redshift=0.1,
        model_name="mosfit_kilonova",
        mag_zpsys="AB",
        dense_resolution=50,
        **kilonova_parameters,
    )

    return KN


def test_kilonova_mag(kilonova_class):
    time = np.array([0.5, 1.0, 2.0])
    mag = kilonova_class.get_apparent_magnitude(time=time, band="lsstr")

    npt.assert_equal(np.shape(mag), np.shape(time))
    npt.assert_(np.all(np.isfinite(mag)))
    npt.assert_(np.all(mag > 0))


def test_kilonova_missing_parameters(kilonova_parameters):
    # Test that omitting a required model parameter raises an error.
    incomplete_parameters = kilonova_parameters.copy()
    incomplete_parameters.pop("mej_1")

    with pytest.raises(TypeError):
        Kilonova(
            redshift=0.1,
            **incomplete_parameters,
        )


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


if __name__ == "__main__":
    pytest.main()
