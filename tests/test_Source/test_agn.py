from slsim.Sources.agn import (
    Agn,
    RandomAgn
)
from astropy import cosmology
import pytest


agn_params = {
    "accretion_disk": "thin_disk",
    "black_hole_mass_exponent": 8.0,
    "black_hole_spin": 0.5,
    "inclination_angle": 30,
    "r_out": 1000,
    "r_resolution": 1000,
    "eddington_ratio": 0.5,
    "corona_height": 10,
}
i_band_mag = 20
redshift = 1
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)


def test_agn_init():
    Agn(i_band_mag, redshift, cosmo=cosmo, **agn_params)

    with pytest.raises(ValueError):
        less_params = {
            "accretion_disk": "thin_disk",
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.5,
            "inclination_angle": 0,
        }

        Agn(i_band_mag, redshift, cosmo=cosmo, **less_params)

    with pytest.raises(ValueError):
        unsupported_disk_kwargs = {
            "accretion_disk": "unsupported_disk",
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.5,
            "inclination_angle": 0,
            "r_out": 1000,
            "r_resolution": 1000,
            "eddington_ratio": 0.1,
        }

        Agn(i_band_mag, redshift, cosmo=cosmo, **unsupported_disk_kwargs)


def test_mean_mags():
    my_agn = Agn(i_band_mag, redshift, cosmo=cosmo, **agn_params)
    survey = "not_lsst"
    with pytest.raises(ValueError):
        my_agn.get_mean_mags(survey)

    survey = "lsst"

    magnitudes = my_agn.get_mean_mags(survey)

    assert magnitudes[0] < magnitudes[-1]
    assert magnitudes[3] == i_band_mag


def test_random_agn():
    
    random_agn_1 = RandomAgn(
        i_band_mag,
        redshift,
        seed=1,
    )
    
    # Make sure the returned AGN has expected method
    lsst_mags_1 = random_agn_1.get_mean_mags('lsst')

    partial_kwargs = {
        "black_hole_mass_exponent": 7.5,
        "eddington_ratio": 0.1,
    }

    random_agn_2 = RandomAgn(
        i_band_mag,
        redshift,
        seed=2,
        **partial_kwargs
    )
    lsst_mags_2 = random_agn_2.get_mean_mags('lsst')

    # Make sure the returned mean mags are different (except for i band)
    assert lsst_mags_1[3] == lsst_mags_2[3]
    for jj in range(3):
        assert lsst_mags_1[jj] != lsst_mags_2[jj]
    for jj in range(2):
        assert lsst_mags_1[jj+4] != lsst_mags_2[jj+4]

    
