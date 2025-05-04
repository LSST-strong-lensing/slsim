import os
import pytest
import numpy as np
import astropy.units as u

# import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

# Import the class to test
from slsim.Microlensing.lightcurve import (
    MicrolensingLightCurve,
)

# Import the MagnificationMap class for a dummy instance
from slsim.Microlensing.magmap import MagnificationMap  # Assuming this exists

# import matplotlib.pyplot as plt

# ---- Test Fixtures ----


@pytest.fixture
def theta_star():
    """Provides a theta_star value for testing."""
    return 4e-6  # arcsec


# Create a dummy MagnificationMap class for isolated testing
@pytest.fixture
def dummy_magmap(theta_star):  # Request theta_star as argument
    """Provides a basic MagnificationMap instance for testing."""

    # Robust path handling (adjust if needed for your structure)
    try:
        # Assuming TestData is in the same directory as the test file
        test_dir = os.path.dirname(__file__)
        magmap2D_path = os.path.join(test_dir, "..", "TestData", "test_magmap2D.npy")
        if not os.path.exists(magmap2D_path):
            # Add more debug info for CI if it fails again
            print(f"[Fixture Debug] Looking for NPY in: {magmap2D_path}")
            print(f"[Fixture Debug] Current working directory: {os.getcwd()}")
            print(
                f"[Fixture Debug] Contents of test directory ({test_dir}): {os.listdir(test_dir)}"
            )
            pytest.fail(f"TestData file not found: {magmap2D_path}")
        magmap2D = np.load(magmap2D_path)
    except Exception as e:
        pytest.fail(f"Failed to load TestData/test_magmap2D.npy: {e}")

    # a precomputed map for the parameters below is available in the TestData folder
    # Use the injected theta_star value
    kwargs_MagnificationMap = {
        "kappa_tot": 0.34960889,
        "shear": 0.34860889,
        "kappa_star": 0.24,
        "theta_star": theta_star,  # Use the fixture value
        "center_x": 0,  # arcsec
        "center_y": 0,  # arcsec
        "half_length_x": 25 * theta_star,  # Use the fixture value
        "half_length_y": 25 * theta_star,  # Use the fixture value
        "mass_function": "kroupa",
        "m_solar": 1.0,
        "m_lower": 0.08,
        "m_upper": 100,
        "num_pixels_x": 1000,
        "num_pixels_y": 1000,
        "kwargs_IPM": {},
    }

    magmap = MagnificationMap(
        magnifications_array=magmap2D,
        **kwargs_MagnificationMap,
    )
    return magmap


# Create a dummy cosmology for testing
@pytest.fixture
def cosmo():
    """Provides a standard cosmology."""
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def time_duration():
    """Provides a time duration in days for testing."""
    return 4000  # days


# create a MicrolensingLightCurve instance for testing
@pytest.fixture
def mlc_instance(dummy_magmap, time_duration):  # Request fixtures as arguments
    """Provides a MicrolensingLightCurve instance."""
    return MicrolensingLightCurve(
        magnification_map=dummy_magmap,  # Use injected value
        time_duration=time_duration,  # Use injected value
    )


@pytest.fixture
def expected_lightcurve_length(mlc_instance, cosmo, source_redshift, source_velocity):
    """Provides an expected light curve length for testing."""
    pixel_size_arcsec = mlc_instance.magnification_map.pixel_size
    pixel_size_kpc = (
        ((cosmo.kpc_proper_per_arcmin(source_redshift)).to(u.kpc / u.arcsec))
        * pixel_size_arcsec
        * u.arcsec
    )
    pixel_size_meter = (pixel_size_kpc.to(u.m)).value
    expected_time_years = mlc_instance.time_duration / 365.25

    # based on velocity find the length of the light curve
    source_velocity_mps = source_velocity * 1000  # convert km/s to m/s
    expected_time_seconds = expected_time_years * 365.25 * 24 * 3600
    # Recalculate expected length using the logic from extract_light_curve
    pixels_traversed = source_velocity_mps * expected_time_seconds / pixel_size_meter
    expected_length = int(pixels_traversed) + 2
    return expected_length  # Return the expected length of the light curve


@pytest.fixture
def source_size():
    """Provides a source size in arcsec for testing."""
    return 8e-8  # arcsec


@pytest.fixture
def source_velocity():
    """Provides a source velocity in km/s for testing."""
    return 1000  # km/s


@pytest.fixture
def source_redshift():
    """Provides a source redshift for testing."""
    return 1.5


# ---- Test Class ----

class TestMicrolensingLightCurve:
    pass