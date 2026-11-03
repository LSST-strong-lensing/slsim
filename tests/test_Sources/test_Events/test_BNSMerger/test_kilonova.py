import numpy as np
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.Events.BNSMerger.kilonova import Kilonova
import numpy.testing as npt
import pytest
from unittest.mock import Mock, patch


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
    return Kilonova(
        redshift=0.1,
        model_name="mosfit_kilonova",
        backend="redback",
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        model_parameters=kilonova_parameters,
        dense_resolution=50,
    )


def test_kilonova_mag(kilonova_class):
    time = np.array([0.5, 1.0, 2.0])
    mag = kilonova_class.get_apparent_magnitude(time=time, band="r")

    npt.assert_equal(np.shape(mag), np.shape(time))
    npt.assert_(np.all(np.isfinite(mag)))
    npt.assert_(np.all(mag > 0))


def test_kilonova_missing_parameters(kilonova_parameters):
    # Test that omitting a required model parameter raises an error.
    incomplete_parameters = kilonova_parameters.copy()
    incomplete_parameters.pop("mej_1")

    kilonova = Kilonova(
        redshift=0.1,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        model_parameters=incomplete_parameters,
    )

    with pytest.raises(TypeError):
        kilonova.get_apparent_magnitude(
            time=np.array([0.5, 1.0]),
            band="r",
        )


def test_kilonova_invalid_model_name(kilonova_parameters):
    # Test that an unavailable Redback model name raises an error.
    with pytest.raises(AttributeError):
        Kilonova(
            redshift=0.1,
            model_name="not_a_kilonova_model",
            **kilonova_parameters,
        )


def test_kilonova_mosfit():
    time = np.array([2.0, 1.0])
    filter_name = "SLSim_Roman_F213"
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Simulate MOSFiT output in a different time order.
    entry = {
        "photometry": [
            {
                "time": "1.0",
                "band": filter_name,
                "magnitude": "23.0",
                "realization": "1",
            },
            {
                "time": "2.0",
                "band": filter_name,
                "magnitude": "24.0",
                "realization": "1",
            },
        ]
    }

    fitter = Mock()
    fitter.fit_events.return_value = [[[entry]]]

    with patch("mosfit.fitter.Fitter", return_value=fitter), patch.object(
        Kilonova,
        "_configure_mosfit_filter",
        return_value=filter_name,
    ):
        kilonova = Kilonova(
            redshift=0.1,
            model_name="bns",
            backend="mosfit",
            cosmo=cosmo,
            model_parameters={"Mchirp": 1.2, "q": 0.9},
        )

        mag = kilonova.get_apparent_magnitude(time=time, band="F213")

    npt.assert_allclose(mag, [24.0, 23.0])

    fitter.fit_events.assert_called_once()
    inputs = fitter.fit_events.call_args.kwargs

    assert inputs["models"] == ["bns"]
    assert inputs["band_list"] == [filter_name]
    assert inputs["band_systems"] == ["AB"]

    fixed = inputs["user_fixed_parameters"]
    parameters = dict(zip(fixed[::2], fixed[1::2]))

    assert float(parameters["Mchirp"]) == 1.2
    assert float(parameters["q"]) == 0.9
    assert float(parameters["redshift"]) == 0.1

    npt.assert_allclose(
        float(parameters["lumdist"]),
        cosmo.luminosity_distance(0.1).to_value("Mpc"),
    )


def test_kilonova_mosfit_runs(tmp_path, monkeypatch):
    # Write temporary filter files into the test directory.
    monkeypatch.chdir(tmp_path)

    kilonova = Kilonova(
        redshift=0.1,
        model_name="bns",
        backend="mosfit",
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        model_parameters={
            "Mchirp": 1.2,
            "q": 0.9,
        },
    )

    time = np.array([0.5, 1.0, 2.0])

    mag = kilonova.get_apparent_magnitude(
        time=time,
        band="F213",
    )

    assert np.shape(mag) == np.shape(time)
    assert np.all(np.isfinite(mag))


if __name__ == "__main__":
    pytest.main()
