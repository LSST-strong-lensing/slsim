import numpy as np
from slsim.slsim.Sources.agn import Agn
from astropy import cosmology
from astropy import units as u
from astropy import constants as const
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

        unsupported_disk_kwargs = {
            "accretion_disk": "MHD_disk",
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
