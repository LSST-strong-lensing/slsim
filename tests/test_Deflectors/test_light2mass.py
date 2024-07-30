#!/usr/bin/env python
import numpy as np
from slsim.Deflectors.light2mass import get_velocity_dispersion
from astropy.cosmology import FlatLambdaCDM
from slsim.Util.mag2errors import get_errors_Poisson
import pytest

def test_get_velocity_dispersion():

    cosmo = FlatLambdaCDM(H0=72, Om0=0.26)
    deflector_type = "galaxy-elliptical"
    lsst_mags = np.array([17.636, 16.674, 16.204]).reshape(1, 3)  # g,,r,i mags
    #lsst_errs = np.array([0.007, 0.005, 0.005]).reshape(1, 3)  # g,r,i errors, if known
    
    # extract errors (due to Poisson noise only) if the errors are not known
    zeropoint_g, exptime_g = 28.51, 15
    zeropoint_r, exptime_r = 28.36, 15
    zeropoint_i, exptime_i = 28.17, 15
    lsst_errs = lsst_errs = np.array([
        get_errors_Poisson(lsst_mags[:,0], zeropoint_g, exptime_g),
        get_errors_Poisson(lsst_mags[:,1], zeropoint_r, exptime_r),
        get_errors_Poisson(lsst_mags[:,2], zeropoint_i, exptime_i),
    ]).T
    redshifts = np.array([0.08496])  # redshift

    coefficients = [
        0.01011,
        0.01920,
        0.05162,
        -0.00032,
        0.06555,
        -0.02949,
        0.00003,
        0.04040,
        -0.00892,
        -0.03068,
        -0.21527,
        0.09394,
    ]  # [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]
    # coefficients = [0,0,0,0,0,0,0,0,0,0,0,0]            #in case input mags are in SDSS bands

    # Get velocity dispersion using spectroscopy based relations
    vel_disp_spec = get_velocity_dispersion(
        deflector_type,
        lsst_mags,
        lsst_errs,
        zz=redshifts,
        cosmo=cosmo,
        bands=["g", "r", "i"],
        c1=coefficients[0],
        c2=coefficients[1],
        c3=coefficients[2],
        c4=coefficients[3],
        c5=coefficients[4],
        c6=coefficients[5],
        c7=coefficients[6],
        c8=coefficients[7],
        c9=coefficients[8],
        c10=coefficients[9],
        c11=coefficients[10],
        c12=coefficients[11],
        scaling_relation="spectroscopic",
    )

    np.testing.assert_equal(len(vel_disp_spec), len(redshifts))
    np.testing.assert_almost_equal(vel_disp_spec[0].nominal_value, 149, decimal=-1)
    # the returned value should be precise within +-10 km/s

    # Get velocity dispersion using weak-lensing based relations
    vel_disp_wl = get_velocity_dispersion(
        deflector_type,
        lsst_mags,
        lsst_errs,
        zz=redshifts,
        cosmo=cosmo,
        bands=["g", "r", "i"],
        c1=coefficients[0],
        c2=coefficients[1],
        c3=coefficients[2],
        c4=coefficients[3],
        c5=coefficients[4],
        c6=coefficients[5],
        c7=coefficients[6],
        c8=coefficients[7],
        c9=coefficients[8],
        c10=coefficients[9],
        c11=coefficients[10],
        c12=coefficients[11],
        scaling_relation="weak-lensing",
    )

    np.testing.assert_equal(len(vel_disp_wl), len(redshifts))
    np.testing.assert_almost_equal(vel_disp_wl[0].nominal_value, 143, decimal=-1)


def test_invalid_deflector_type():

    with pytest.raises(KeyError, match="The module currently supports only elliptical galaxies."):
        get_velocity_dispersion(deflector_type="galaxy-spiral",
                                lsst_mags=np.array([17.636, 16.674, 16.204]).reshape(1, 3),
                                lsst_errs=np.array([0.007, 0.005, 0.005]).reshape(1, 3),
                                zz=np.array([0.08496]),
                                cosmo=FlatLambdaCDM(H0=72, Om0=0.26),
                                scaling_relation="spectroscopic")


def test_invalid_scaling_relations():
    
    with pytest.raises(KeyError, match="Invalid input for scaling relations."):

        get_velocity_dispersion(deflector_type="galaxy-elliptical",
                                lsst_mags=np.array([17.636, 16.674, 16.204]).reshape(1, 3),
                                lsst_errs=np.array([0.007, 0.005, 0.005]).reshape(1, 3),
                                zz=np.array([0.08496]),
                                cosmo=FlatLambdaCDM(H0=72, Om0=0.26),
                                scaling_relation="xyz")


if __name__ == "__main__":
    test_get_velocity_dispersion()
    test_invalid_deflector_type()
    test_invalid_scaling_relations()
