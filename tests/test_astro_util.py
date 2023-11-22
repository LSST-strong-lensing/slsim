import numpy as np
from numpy import testing as npt
import pytest
from astropy import constants as const
from slsim.Util.astro_util import (
    spin_to_isco,
    calculate_eddington_luminosity,
    eddington_ratio_to_accreted_mass,
)


def test_spin_to_isco():
    # Check all special cases of
    # spin = 0, the Schwarzschild black hole
    # spin = 1, maximum prograde Kerr black hole
    # spin = -1, maximum retrograde Kerr black hole
    assert spin_to_isco(0.0) == 6.0
    assert spin_to_isco(-1.0) == 9.0
    assert spin_to_isco(1.0) == 1.0
    # test errors
    with pytest.raises(ValueError):
        spin_to_isco(2.0)
        spin_to_isco(-100.0)


def test_calculate_eddington_luminosity():
    # Check some cases
    test_value_1 = (
        4 * np.pi * const.G * const.m_p * const.c * const.M_sun / const.sigma_T
    )
    npt.assert_approx_equal(calculate_eddington_luminosity(0).value, test_value_1.value)

    test_value_2 = (
        4
        * np.pi
        * const.G
        * const.m_p
        * 100000000
        * const.c
        * const.M_sun
        / const.sigma_T
    )
    npt.assert_approx_equal(calculate_eddington_luminosity(8).value, test_value_2.value)
    assert calculate_eddington_luminosity(8).unit == test_value_2.unit


def test_eddington_ratio_to_accreted_mass():
    # Test zero efficiency error
    with pytest.raises(ValueError):
        eddington_ratio_to_accreted_mass(8.0, 0.1, efficiency=0)

    # Test zero eddington ratio
    assert eddington_ratio_to_accreted_mass(8.0, 0.0) == 0

    # Test typical value
    expected_value = calculate_eddington_luminosity(8.0) * 0.1 / (0.1 * const.c**2)

    npt.assert_approx_equal(
        eddington_ratio_to_accreted_mass(8.0, 0.1).value, expected_value.value
    )
    assert eddington_ratio_to_accreted_mass(8.0, 0.1).unit == expected_value.unit
