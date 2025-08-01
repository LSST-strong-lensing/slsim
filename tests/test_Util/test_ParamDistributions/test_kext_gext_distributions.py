from slsim.Util.ParamDistributions.kext_gext_distributions import (
    LineOfSightDistribution,
)
import pytest
import os


@pytest.fixture(autouse=True)
def reset_line_of_sight_data():
    LineOfSightDistribution.correction_data = None
    LineOfSightDistribution.no_nonlinear_correction_data = None


def test_initializes_with_default_data_paths():
    line_of_sight = LineOfSightDistribution()
    # assert line_of_sight.no_nonlinear_correction_data is not None

    get_kappa_gamma = line_of_sight.get_kappa_gamma(
        0.5, 0.3, use_nonlinear_correction=False
    )
    assert isinstance(get_kappa_gamma, tuple)
    assert len(get_kappa_gamma) == 2

    get_kappa_gamma = line_of_sight.get_kappa_gamma(
        0.5, 0.3, use_nonlinear_correction=True
    )
    assert isinstance(get_kappa_gamma, tuple)
    assert len(get_kappa_gamma) == 2


def test_round_to_nearest_0_1_multiple_decimal_places():
    los_distribution = LineOfSightDistribution()
    result = los_distribution._round_to_nearest_0_1(2.345)
    assert result == 2.3
    result2 = los_distribution._round_to_nearest_0_1(5.6)
    assert result2 == 4.9
    result3 = los_distribution._round_to_nearest_0_1(0.05)
    assert result3 == 0.1


def test_valid_inputs():
    line_of_sight = LineOfSightDistribution()
    result = line_of_sight.get_kappa_gamma(0.5, 0.3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)

    result2 = line_of_sight.get_kappa_gamma(0.48, 0.49)
    assert isinstance(result2, tuple)
    assert len(result2) == 2
    assert isinstance(result2[0], float)
    assert isinstance(result2[1], float)

    result3 = line_of_sight.get_kappa_gamma(0.3, 0.4)
    assert isinstance(result3, tuple)
    assert len(result3) == 2
    assert result3[0] == 0.0


def test_invalid():
    with pytest.raises(FileNotFoundError):
        LineOfSightDistribution(
            nonlinear_correction_path="wrong", no_correction_path="wrong"
        )
    with pytest.raises(FileNotFoundError):
        LineOfSightDistribution(no_correction_path="wrong")

    current_script_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_script_path)
    parent_directory = os.path.dirname(os.path.dirname(current_directory))
    file_path = os.path.join(parent_directory, "TestData/empty_file.h5")

    los2 = LineOfSightDistribution(
        nonlinear_correction_path=file_path, no_correction_path=file_path
    )
    assert los2.correction_data == {}
    assert los2.no_nonlinear_correction_data == {}

    with pytest.raises(ValueError):
        los2.get_kappa_gamma(0.5, 0.3)

    assert los2.correction_data == {}
    assert los2.no_nonlinear_correction_data == {}


@pytest.fixture(autouse=True)
def reset_line_of_sight():
    LineOfSightDistribution.correction_data = None
    LineOfSightDistribution.no_nonlinear_correction_data = None

    yield

    LineOfSightDistribution.correction_data = None
    LineOfSightDistribution.no_nonlinear_correction_data = None
