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


def test_random_agn_variability_from_distribution():
    # Setup for the new distribution-based variability model

    valid_band = "lsst2016-i"  # Used for valid tests
    invalid_band_for_defaults = (
        "lsst2023-r"  # Used to trigger errors when defaults are attempted
    )

    variability_model = "bending_power_law_from_distribution"

    # Mock means and covs (dimensions must match expectations of underlying util functions,
    # i.e. 5 params: log_BH_mass, M_i, log_SFi_inf, log_tau, zsrc)
    means = np.array([8.0, -22.0, -0.5, 2.5, 1.0])
    covs = np.identity(5) * 0.1

    kwargs_variability_valid = {
        "multivariate_gaussian_means": means,
        "multivariate_gaussian_covs": covs,
        "known_band": valid_band,
    }

    # Test 1: Raise ValueError if known_band in kwargs does not match known_band argument
    with pytest.raises(ValueError) as excinfo:
        RandomAgn(
            known_band="lsst2023-r",
            known_mag=20,
            redshift=1.0,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=variability_model,
            agn_driving_kwargs_variability=kwargs_variability_valid,
            black_hole_mass_exponent=8.0,
        )
    assert "known_band in agn_driving_kwargs_variability does not match" in str(
        excinfo.value
    )

    # Test 2: Raise ValueError if multivariate_gaussian_means is missing AND band is not supported for defaults.
    # (Triggers default logic -> Fails because 'r' band is not 'i' band)
    with pytest.raises(ValueError) as excinfo:
        RandomAgn(
            known_band=invalid_band_for_defaults,
            known_mag=20,
            redshift=1.0,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=variability_model,
            agn_driving_kwargs_variability={
                "multivariate_gaussian_covs": covs,
                "known_band": invalid_band_for_defaults,
            },
            black_hole_mass_exponent=8.0,
        )
    assert "known_band in kwargs_agn_model must be lsst2016-i or lsst2023-i" in str(
        excinfo.value
    )

    # Test 3: Raise ValueError if multivariate_gaussian_covs is missing AND band is not supported for defaults.
    # (Triggers default logic -> Fails because 'r' band is not 'i' band)
    with pytest.raises(ValueError) as excinfo:
        RandomAgn(
            known_band=invalid_band_for_defaults,
            known_mag=20,
            redshift=1.0,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=variability_model,
            agn_driving_kwargs_variability={
                "multivariate_gaussian_means": means,
                "known_band": invalid_band_for_defaults,
            },
            black_hole_mass_exponent=8.0,
        )
    assert "known_band in kwargs_agn_model must be lsst2016-i or lsst2023-i" in str(
        excinfo.value
    )

    # Test 4: Raise ValueError if known_band is missing in kwargs but means/covs ARE present.
    with pytest.raises(ValueError) as excinfo:
        RandomAgn(
            known_band=valid_band,
            known_mag=20,
            redshift=1.0,
            lightcurve_time=lightcurve_time,
            agn_driving_variability_model=variability_model,
            agn_driving_kwargs_variability={
                "multivariate_gaussian_means": means,
                "multivariate_gaussian_covs": covs,
            },
            black_hole_mass_exponent=8.0,
        )
    assert (
        "known_band not found in agn_driving_kwargs_variability when multivariate"
        in str(excinfo.value)
    )

    # Test 5: Successful generation with EXPLICIT means/covs
    random_agn_explicit = RandomAgn(
        known_band=valid_band,
        known_mag=20,
        redshift=1.0,
        lightcurve_time=None,
        agn_driving_variability_model=variability_model,
        agn_driving_kwargs_variability=kwargs_variability_valid,
        black_hole_mass_exponent=8.0,
        random_seed=42,
    )

    # Verify that the model was converted internally from "bending_power_law_from_distribution"
    # to "bending_power_law"
    assert random_agn_explicit.agn_driving_variability_model == "bending_power_law"

    # Verify that the driving kwargs now contain the generated parameters
    explicit_kwargs = random_agn_explicit.agn_driving_kwargs_variability
    assert "log_breakpoint_frequency" in explicit_kwargs
    assert "standard_deviation" in explicit_kwargs
    assert "length_of_light_curve" in explicit_kwargs
    assert explicit_kwargs["low_frequency_slope"] == 0  # DRW default
    assert explicit_kwargs["high_frequency_slope"] == 2  # DRW default

    # Verify time array was generated (default length is 1000)
    assert len(random_agn_explicit.kwargs_model["time_array"]) == 1000
    assert explicit_kwargs["length_of_light_curve"] == 999.0

    # Test 6: Successful generation with DEFAULTS (MacLeod 2010)
    # Passing empty kwargs + i-band should trigger the default parameters.
    random_agn_defaults = RandomAgn(
        known_band="lsst2023-i",
        known_mag=20,
        redshift=1.0,
        lightcurve_time=None,
        agn_driving_variability_model=variability_model,
        agn_driving_kwargs_variability={},  # Empty dict triggers defaults
        black_hole_mass_exponent=8.0,
        random_seed=42,
    )

    # Assertions for Default Case
    assert random_agn_defaults.agn_driving_variability_model == "bending_power_law"
    default_kwargs = random_agn_defaults.agn_driving_kwargs_variability

    # Verify the calculated parameters exist
    assert "log_breakpoint_frequency" in default_kwargs
    assert "standard_deviation" in default_kwargs
    assert "length_of_light_curve" in default_kwargs
    assert default_kwargs["low_frequency_slope"] == 0
    assert default_kwargs["high_frequency_slope"] == 2
