#!/usr/bin/env python
import numpy as np
from slsim.Deflectors.light2mass import get_velocity_dispersion
from astropy.cosmology import FlatLambdaCDM
from slsim.Util.mag2errors import get_errors_Poisson
import pytest


def test_get_velocity_dispersion():
    """Test velocity dispersion for both 5-band and 3-band inputs."""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector_type = "elliptical"

    # --- 5-band spectroscopic & weak-lensing ---
    lsst_mags_5 = np.array([19.492, 17.636, 16.674, 16.204, 15.893]).reshape(1, 5)
    zeropoints = [26.52, 28.51, 28.36, 28.17, 27.78]
    exptimes = [15, 15, 15, 15, 15]
    errs_5 = np.array(
        [
            get_errors_Poisson(lsst_mags_5[:, i], zeropoints[i], exptimes[i])
            for i in range(5)
        ]
    ).T  # errs_5 will be (1,5)
    redshifts = np.array([0.08496])

    vel_disp_spec_5 = get_velocity_dispersion(
        deflector_type,
        lsst_mags_5,
        errs_5,
        redshift=redshifts,
        cosmo=cosmo,
        bands=["u", "g", "r", "i", "z"],
        scaling_relation="spectroscopic",
    )
    # Expected value ~203 km/s (within ±10 km/s)
    np.testing.assert_almost_equal(vel_disp_spec_5[0].nominal_value, 203, decimal=-1)

    vel_disp_wl_5 = get_velocity_dispersion(
        deflector_type,
        lsst_mags_5,
        errs_5,
        redshift=redshifts,
        cosmo=cosmo,
        bands=["u", "g", "r", "i", "z"],
        scaling_relation="weak-lensing",
    )
    np.testing.assert_almost_equal(vel_disp_wl_5[0].nominal_value, 182, decimal=-1)

    # --- 3-band spectroscopic & weak-lensing ---
    lsst_mags_3 = np.array([17.636, 16.674, 16.204]).reshape(1, 3)
    zeropoints_3 = [28.51, 28.36, 28.17]
    exptimes_3 = [15, 15, 15]
    errs_3 = np.array(
        [
            get_errors_Poisson(lsst_mags_3[:, i], zeropoints_3[i], exptimes_3[i])
            for i in range(3)
        ]
    ).T  # errs_3 is (1,3)

    vel_disp_spec_3 = get_velocity_dispersion(
        deflector_type,
        lsst_mags_3,
        errs_3,
        redshift=redshifts,
        cosmo=cosmo,
        bands=["g", "r", "i"],
        scaling_relation="spectroscopic",
    )
    # Expected value ~203 km/s (within ±10% tolerance)
    np.testing.assert_allclose(vel_disp_spec_3[0].nominal_value, 203, rtol=0.10)
    assert isinstance(vel_disp_spec_3, np.ndarray)
    assert vel_disp_spec_3.shape == (1,)

    vel_disp_wl_3 = get_velocity_dispersion(
        deflector_type,
        lsst_mags_3,
        errs_3,
        redshift=redshifts,
        cosmo=cosmo,
        bands=["g", "r", "i"],
        scaling_relation="weak-lensing",
    )
    np.testing.assert_almost_equal(vel_disp_wl_3[0].nominal_value, 182, decimal=-1)
    assert isinstance(vel_disp_wl_3, np.ndarray)
    assert vel_disp_wl_3.shape == (1,)


def test_invalid_deflector_type():
    with pytest.raises(
        KeyError, match="The module currently supports only elliptical galaxies."
    ):
        get_velocity_dispersion(
            deflector_type="spiral",
            lsst_mags=np.array([19.492, 17.636, 16.674, 16.204, 15.893]).reshape(1, 5),
            lsst_errs=np.array([0.052, 0.007, 0.005, 0.005, 0.01]).reshape(1, 5),
            redshift=np.array([0.08496]),
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
            scaling_relation="spectroscopic",
        )

def test_missing_required_bands():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector_type = "elliptical"

    lsst_mags_3 = np.array([17.636, 16.674, 16.204]).reshape(1, 3)
    zeropoints_3 = [28.51, 28.36, 28.17]
    exptimes_3 = [15, 15, 15]
    errs_3 = np.array(
        [
            get_errors_Poisson(lsst_mags_3[:, i], zeropoints_3[i], exptimes_3[i])
            for i in range(3)
        ]
    ).T 
    redshifts = np.array([0.08496])
 
    with pytest.raises(ValueError, match="input requires at least g r and i band"):
        get_velocity_dispersion(
            deflector_type,
            lsst_mags_3,
            errs_3,
            redshift=redshifts,
            cosmo=cosmo,
            bands=["g", "r"],
            scaling_relation="spectroscopic",
        )

if __name__ == "__main__":
    test_get_velocity_dispersion()
    test_invalid_deflector_type()
    test_missing_required_bands()