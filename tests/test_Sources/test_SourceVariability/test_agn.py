import numpy as np
from slsim.Sources.SourceVariability.agn import Agn, RandomAgn
from astropy import cosmology
import pytest


# Define some parameters common to all agn in this test
# These are the intrinsic signal I am choosing
agn_driving_variability_model = "light_curve"
agn_driving_kwargs_variability = {
    "MJD": np.linspace(1, 500, 500),
    "ps_mag_intrinsic": 10 + np.sin(np.linspace(1, 500, 500) * np.pi / 30),
}

# These are observational time stamps (note they do not have to line up with input signal)
lightcurve_time = np.linspace(1, 500, 20)

# General agn parameters held in a dictionary
agn_params = {
    "accretion_disk": "thin_disk",
    "black_hole_mass_exponent": 8.0,
    "black_hole_spin": 0.5,
    "inclination_angle": 30,
    "r_out": 1000,
    "r_resolution": 1000,
    "eddington_ratio": 0.5,
    "corona_height": 10,
    "speclite_filter": "lsst2023-i",
}

# Define the known magnitude (will be eventually input from source.source_dict["ps_mag_i"] column)
i_band_string = "lsst2023-i"
i_band_mag = 20

# Other cosmological params of the source
redshift = 1
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)


def test_agn_init():
    Agn(
        i_band_string,
        i_band_mag,
        redshift,
        cosmo=cosmo,
        lightcurve_time=lightcurve_time,
        agn_driving_variability_model=agn_driving_variability_model,
        agn_driving_kwargs_variability=agn_driving_kwargs_variability,
        **agn_params
    )

    with pytest.raises(ValueError):
        less_params = {
            "accretion_disk": "thin_disk",
            "black_hole_mass_exponent": 8.0,
            "black_hole_spin": 0.5,
            "inclination_angle": 0,
            "driving_variability": "intrinsic_light_curve",
        }

        Agn(
            i_band_string,
            i_band_mag,
            redshift,
            cosmo=cosmo,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=agn_driving_variability_model,
            agn_driving_kwargs_variability=agn_driving_kwargs_variability,
            **less_params
        )

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

        Agn(
            i_band_string,
            i_band_mag,
            redshift,
            cosmo=cosmo,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=agn_driving_variability_model,
            agn_driving_kwargs_variability=agn_driving_kwargs_variability,
            **unsupported_disk_kwargs
        )

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

        Agn(
            i_band_string,
            i_band_mag,
            redshift,
            cosmo=cosmo,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=agn_driving_variability_model,
            agn_driving_kwargs_variability=agn_driving_kwargs_variability,
            **unsupported_driving_kwargs
        )


def test_mean_mags():

    my_agn = Agn(
        i_band_string,
        i_band_mag,
        redshift,
        cosmo=cosmo,
        lightcurve_time=lightcurve_time,
        agn_driving_variability_model=agn_driving_variability_model,
        agn_driving_kwargs_variability=agn_driving_kwargs_variability,
        **agn_params
    )

    good_bands = [
        "lsst2023-u",
        "lsst2023-i",
        "lsst2016-z",
        "lsst2016-y",
    ]

    magnitudes = my_agn.get_mean_mags(good_bands)

    assert magnitudes[0] != magnitudes[-1]
    assert magnitudes[1] == i_band_mag


def test_random_agn():

    no_kwargs = {}

    random_agn_1 = RandomAgn(
        i_band_string,
        i_band_mag,
        redshift,
        lightcurve_time=lightcurve_time,
        agn_driving_variability_model=agn_driving_variability_model,
        agn_driving_kwargs_variability=agn_driving_kwargs_variability,
        random_seed=1,
        **no_kwargs
    )

    # Check generating with only a few kwargs
    partial_kwargs = {
        "black_hole_mass_exponent": 7.5,
        "eddington_ratio": 0.1,
    }
    RandomAgn(
        i_band_string,
        i_band_mag,
        redshift,
        lightcurve_time=lightcurve_time,
        agn_driving_variability_model=agn_driving_variability_model,
        agn_driving_kwargs_variability=agn_driving_kwargs_variability,
        random_seed=2,
        **partial_kwargs
    )

    # Check that we can define a dictionary for the boundaries
    # to pull random values from

    input_agn_bounds_dict = {
        "black_hole_mass_exponent_bounds": [8.0, 8.0],
        "black_hole_spin_bounds": [-0.1, 0.1],
        "inclination_angle_bounds": [0, 45],
        "eddington_ratio_bounds": [0.1, 0.1],
        "supported_disk_models": ["thin_disk"],
        "driving_variability": ["intrinsic_light_curve"],
    }

    # source object has most variables inside a Column object,
    # so test that this is compatible
    # dictionary_in_column = Column(data=[input_agn_bounds_dict], name=("dictionary"))

    random_agn_2 = RandomAgn(
        i_band_string,
        i_band_mag,
        redshift,
        random_seed=1,
        lightcurve_time=lightcurve_time,
        agn_driving_variability_model=agn_driving_variability_model,
        agn_driving_kwargs_variability=agn_driving_kwargs_variability,
        input_agn_bounds_dict=input_agn_bounds_dict,
    )

    light_curve_1 = {
        "MJD": np.asarray([0, 1, 2, 3, 4, 5]),
        "ps_mag_intrinsic": np.asarray([1, 0, -1, 0, 1, 0]),
    }
    light_curve_2 = {
        "MJD": np.asarray([0, 1, 2, 3, 4, 5]),
        "ps_mag_intrinsic": np.asarray([1, 0, 1, 0, 1, 0]),
    }
    input_agn_bounds_dict["intrinsic_light_curve"] = [light_curve_1, light_curve_2]

    random_agn_3 = RandomAgn(
        i_band_string,
        i_band_mag,
        redshift,
        random_seed=1,
        lightcurve_time=lightcurve_time,
        input_agn_bounds_dict=input_agn_bounds_dict,
    )

    # test initialization with minimal information
    RandomAgn(i_band_string, i_band_mag, redshift)

    # Test that we raise a warning in RandomAgn when no time axis is input
    with pytest.raises(ValueError):
        RandomAgn(
            i_band_string,
            i_band_mag,
            redshift,
            random_seed=1,
            agn_driving_variability_model=agn_driving_variability_model,
            agn_driving_kwargs_variability=agn_driving_kwargs_variability,
            input_agn_bounds_dict=input_agn_bounds_dict,
        )

    # check that a random value from the range [8.0, 8.0) must return 8.0
    assert random_agn_2.kwargs_model["black_hole_mass_exponent"] == 8.0
    assert random_agn_3.kwargs_model["black_hole_mass_exponent"] == 8.0

    # check the inclination is on range [0, 45)
    assert random_agn_2.kwargs_model["inclination_angle"] < 45.1

    lsst_bands = [
        "lsst2023-u",
        "lsst2023-g",
        "lsst2023-r",
        "lsst2023-i",
        "lsst2023-z",
        "lsst2023-y",
    ]

    # Make sure the returned AGN has expected method
    lsst_mags_1 = random_agn_1.get_mean_mags(lsst_bands)
    lsst_mags_2 = random_agn_2.get_mean_mags(lsst_bands)

    # Make sure the returned mean mags are different (except for i band)
    assert lsst_mags_1[3] == lsst_mags_2[3]
    for jj in range(3):
        assert lsst_mags_1[jj] != lsst_mags_2[jj]
    for jj in range(2):
        assert lsst_mags_1[jj + 4] != lsst_mags_2[jj + 4]
