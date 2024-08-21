import numpy as np
from slsim.Sources.agn import Agn, RandomAgn
from astropy import cosmology
import pytest
from astropy.table import Column

intrinsic_light_curve = {
    "MJD": np.linspace(1, 500, 500),
    "ps_mag_intrinsic": 10 + np.sin(np.linspace(1, 500, 500) * np.pi / 30),
}

agn_params = {
    "accretion_disk": "thin_disk",
    "black_hole_mass_exponent": 8.0,
    "black_hole_spin": 0.5,
    "inclination_angle": 30,
    "r_out": 1000,
    "r_resolution": 1000,
    "eddington_ratio": 0.5,
    "corona_height": 10,
    "driving_variability": "intrinsic_light_curve",
    "intrinsic_light_curve": intrinsic_light_curve,
    "speclite_filter": "lsst2023-i",
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
            "driving_variability": "intrinsic_light_curve",
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
            "driving_variability": "intrinsic_light_curve",
        }

        Agn(i_band_mag, redshift, cosmo=cosmo, **unsupported_disk_kwargs)

    with pytest.raises(ValueError):
        unsupported_driving_kwargs = {
            "accretion_disk": "thin_disk",
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.5,
            "inclination_angle": 0,
            "r_out": 1000,
            "r_resolution": 1000,
            "eddington_ratio": 0.1,
            "driving_variability": "unsupported_variability",
        }

        Agn(i_band_mag, redshift, cosmo=cosmo, **unsupported_driving_kwargs)


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
        random_seed=1,
    )

    # Check generating with only a few kwargs
    partial_kwargs = {
        "black_hole_mass_exponent": 7.5,
        "eddington_ratio": 0.1,
    }
    RandomAgn(i_band_mag, redshift, seed=2, **partial_kwargs)

    # Check that we can define a dictionary for the boundaries
    # to pull random values from

    input_agn_bounds_dict = {
        "black_hole_mass_exponent_bounds": [8.0, 8.0],
        "black_hole_spin_bounds": [-0.1, 0.1],
        "inclination_angle_bounds": [0, 45],
        "r_out_bounds": [1000, 1000],
        "eddington_ratio_bounds": [0.1, 0.1],
        "supported_disk_models": ["thin_disk"],
        "driving_variability": ["intrinsic_light_curve"],
    }

    # source object has most variables inside a Column object,
    # so test that this is compatible
    dictionary_in_column = Column(data=[input_agn_bounds_dict], name=("dictionary"))

    random_agn_2 = RandomAgn(
        i_band_mag, redshift, random_seed=1, input_agn_bounds_dict=input_agn_bounds_dict
    )

    random_agn_2_table = RandomAgn(
        i_band_mag, redshift, random_seed=1, input_agn_bounds_dict=dictionary_in_column
    )

    random_agn_2_ndarray = RandomAgn(
        i_band_mag,
        redshift,
        random_seed=1,
        input_agn_bounds_dict=dictionary_in_column.data,
    )

    # check that a random value from the range [8.0, 8.0) must return 8.0
    assert random_agn_2.kwargs_model["black_hole_mass_exponent"] == 8.0

    # check the inclination is on range [0, 45)
    assert random_agn_2_table.kwargs_model["inclination_angle"] < 45.1

    # Make sure the returned AGN has expected method
    lsst_mags_1 = random_agn_1.get_mean_mags("lsst")

    lsst_mags_2 = random_agn_2.get_mean_mags("lsst")
    lsst_mags_2a = random_agn_2_table.get_mean_mags("lsst")
    lsst_mags_2b = random_agn_2_ndarray.get_mean_mags("lsst")

    # Make sure the returned mean mags are different (except for i band)
    assert lsst_mags_1[3] == lsst_mags_2[3]
    for jj in range(3):
        assert lsst_mags_1[jj] != lsst_mags_2[jj]
    for jj in range(2):
        assert lsst_mags_1[jj + 4] != lsst_mags_2[jj + 4]

    # Make sure mean mags are the same for each method of random_agn_2
    # Each of these should be identical
    for jj in range(6):
        assert lsst_mags_2[jj] == lsst_mags_2a[jj]
        assert lsst_mags_2[jj] == lsst_mags_2b[jj]
