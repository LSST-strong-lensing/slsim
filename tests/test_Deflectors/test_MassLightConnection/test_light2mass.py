#!/usr/bin/env python
import numpy as np
from slsim.Deflectors.MassLightConnection.light2mass import get_velocity_dispersion
from astropy.cosmology import FlatLambdaCDM
from slsim.Util.mag2errors import get_errors_Poisson
import pytest


def test_get_velocity_dispersion():

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector_type = "elliptical"
    lsst_mags = np.array([19.492, 17.636, 16.674, 16.204, 15.893]).reshape(1, 5)

    # extract errors (due to Poisson noise only) if the errors are not known
    zeropoint_u, exptime_u = 26.52, 15
    zeropoint_g, exptime_g = 28.51, 15
    zeropoint_r, exptime_r = 28.36, 15
    zeropoint_i, exptime_i = 28.17, 15
    zeropoint_z, exptime_z = 27.78, 15
    lsst_errs = np.array(
        [
            get_errors_Poisson(lsst_mags[:, 0], zeropoint_u, exptime_u),
            get_errors_Poisson(lsst_mags[:, 1], zeropoint_g, exptime_g),
            get_errors_Poisson(lsst_mags[:, 2], zeropoint_r, exptime_r),
            get_errors_Poisson(lsst_mags[:, 3], zeropoint_i, exptime_i),
            get_errors_Poisson(lsst_mags[:, 4], zeropoint_z, exptime_z),
        ]
    ).T
    redshifts = np.array([0.08496])  # redshift

    # Get velocity dispersion using spectroscopy based relations
    vel_disp_spec = get_velocity_dispersion(
        deflector_type,
        lsst_mags.T,
        lsst_errs.T,
        redshift=redshifts,
        cosmo=cosmo,
        bands=["u", "g", "r", "i", "z"],
        scaling_relation="spectroscopic",
    )
    print(vel_disp_spec[0].nominal_value)

    np.testing.assert_almost_equal(vel_disp_spec[0].nominal_value, 203, decimal=-1)
    # np.testing.assert_almost_equal(vel_disp_spec[0].nominal_value, 179, decimal=-1)
    # the returned value should be precise within +-10 km/s

    # Get velocity dispersion using weak-lensing based relations
    vel_disp_wl = get_velocity_dispersion(
        deflector_type,
        lsst_mags.T,
        lsst_errs.T,
        redshift=redshifts,
        cosmo=cosmo,
        bands=["u", "g", "r", "i", "z"],
        scaling_relation="weak-lensing",
    )
    print(vel_disp_wl[0].nominal_value)

    np.testing.assert_almost_equal(vel_disp_wl[0].nominal_value, 182, decimal=-1)


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


if __name__ == "__main__":
    test_get_velocity_dispersion()
    test_invalid_deflector_type()
