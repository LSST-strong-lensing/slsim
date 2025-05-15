import os
import pytest
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

# Import the class to test
from slsim.Microlensing.magmap import MagnificationMap

# Import dependencies directly as they are assumed available


# ---- Test Fixtures ----


@pytest.fixture(scope="module")
def cosmology():
    """Provides a cosmology instance for testing."""
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture(scope="module")
def magmap_params():
    """Provides the parameters associated with the test magnification map."""
    # These parameters should correspond to the saved TestData/test_magmap2D.npy
    theta_star = (
        1.4533388875267387e-06  # Example value, ensure consistency if map depends on it
    )
    return {
        "kappa_tot": 0.47128266,
        "shear": 0.42394672,
        "kappa_star": 0.12007537,
        "theta_star": theta_star,
        "center_x": 0.0,  # arcsec
        "center_y": 0.0,  # arcsec
        "half_length_x": 2.5 * theta_star,
        "half_length_y": 2.5 * theta_star,
        "mass_function": "kroupa",  # Default, but set explicitly for clarity
        "m_solar": 1.0,
        "m_lower": 0.01,
        "m_upper": 5,
        # These MUST match the dimensions of the loaded magmap_0.npy
        "num_pixels_x": 50,
        "num_pixels_y": 50,
        "kwargs_IPM": {},
    }


@pytest.fixture(scope="module")
def loaded_mag_array():
    """Loads the magnification map data from the test file."""
    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        # Try path relative to test file first
        magmap2D_path = os.path.join(
            test_dir, "..", "TestData", "test_magmaps_microlensing", "magmap_0.npy"
        )

        magmap2D = np.load(magmap2D_path)
        return magmap2D
    except Exception as e:
        pytest.fail(
            f"Failed to load TestData/test_magmaps_microlensing/magmap_0.npy: {e}"
        )


@pytest.fixture
def magmap_instance(loaded_mag_array, magmap_params):
    """Provides a MagnificationMap instance initialized with the loaded array
    and corresponding parameters."""
    # Ensure dimensions in params match loaded array
    params = magmap_params.copy()
    params["num_pixels_y"], params["num_pixels_x"] = (
        loaded_mag_array.shape
    )  # numpy shape is (rows, cols) -> (y, x)

    # Create instance WITH the array, bypassing generation
    magmap = MagnificationMap(
        magnifications_array=loaded_mag_array,
        **params,
    )
    return magmap


# ---- Test Class ----


# Filter potential RuntimeWarnings from log10 in magnitude calculation
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log10:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in log10:RuntimeWarning")
class TestMagnificationMap:

    def test_init_with_array(self, magmap_instance, loaded_mag_array, magmap_params):
        """Tests initialization when providing a magnification array."""
        assert magmap_instance.magnifications is loaded_mag_array
        assert magmap_instance._kappa_tot == magmap_params["kappa_tot"]
        assert magmap_instance._shear == magmap_params["shear"]
        assert magmap_instance._kappa_star == magmap_params["kappa_star"]
        assert magmap_instance.theta_star == magmap_params["theta_star"]
        assert magmap_instance.center_x == magmap_params["center_x"]
        assert magmap_instance.center_y == magmap_params["center_y"]
        assert magmap_instance.half_length_x == magmap_params["half_length_x"]
        assert magmap_instance.half_length_y == magmap_params["half_length_y"]
        assert magmap_instance._mass_function == magmap_params["mass_function"]
        assert magmap_instance._m_solar == magmap_params["m_solar"]
        assert magmap_instance._m_lower == magmap_params["m_lower"]
        assert magmap_instance._m_upper == magmap_params["m_upper"]
        assert magmap_instance.num_pixels_x == loaded_mag_array.shape[1]
        assert magmap_instance.num_pixels_y == loaded_mag_array.shape[0]

    def test_init_defaults(self):
        """Tests that defaults are set when not provided (using dummy
        array)."""
        dummy_array = np.ones((10, 10))
        # Minimal required args if providing array
        magmap = MagnificationMap(magnifications_array=dummy_array)
        assert magmap._mass_function == "kroupa"
        assert magmap._m_solar == 1
        assert magmap._m_lower == 0.08
        assert magmap._m_upper == 100

    # Test properties
    def test_property_mu_ave(self, magmap_instance, magmap_params):
        """Tests the mu_ave property calculation."""
        expected_mu_ave = 1 / (
            (1 - magmap_params["kappa_tot"]) ** 2 - magmap_params["shear"] ** 2
        )
        np.testing.assert_allclose(magmap_instance.mu_ave, expected_mu_ave)

    def test_property_fractions(self, magmap_instance, magmap_params):
        """Tests stellar_fraction and smooth_fraction properties."""
        expected_stellar_frac = magmap_params["kappa_star"] / magmap_params["kappa_tot"]
        expected_smooth_frac = 1.0 - expected_stellar_frac
        np.testing.assert_allclose(
            magmap_instance.stellar_fraction, expected_stellar_frac
        )
        np.testing.assert_allclose(
            magmap_instance.smooth_fraction, expected_smooth_frac
        )

    def test_property_num_pixels(self, magmap_instance, loaded_mag_array):
        """Tests the num_pixels property."""
        expected_num_pix = (loaded_mag_array.shape[1], loaded_mag_array.shape[0])
        assert magmap_instance.num_pixels == expected_num_pix

    def test_property_pixel_scales(self, magmap_instance, magmap_params):
        """Tests the pixel_scales property."""
        expected_scale_x = (
            2 * magmap_params["half_length_x"] / magmap_params["num_pixels_x"]
        )
        expected_scale_y = (
            2 * magmap_params["half_length_y"] / magmap_params["num_pixels_y"]
        )
        assert isinstance(magmap_instance.pixel_scales, tuple)
        assert len(magmap_instance.pixel_scales) == 2
        np.testing.assert_allclose(magmap_instance.pixel_scales[0], expected_scale_x)
        np.testing.assert_allclose(magmap_instance.pixel_scales[1], expected_scale_y)

    def test_property_pixel_size(self, magmap_instance):
        """Tests the pixel_size property."""
        scale_x, scale_y = magmap_instance.pixel_scales
        expected_pixel_size = np.sqrt(scale_x * scale_y)
        np.testing.assert_allclose(magmap_instance.pixel_size, expected_pixel_size)

    def test_property_magnitudes(self, magmap_instance, loaded_mag_array):
        """Tests the magnitudes property calculation."""
        mu_ave = magmap_instance.mu_ave
        magnifications = loaded_mag_array
        valid_mask = (
            (magnifications > 0) & np.isfinite(magnifications) & (np.abs(mu_ave) > 0)
        )
        expected_mags = np.full_like(magnifications, np.nan, dtype=float)
        expected_mags[valid_mask] = -2.5 * np.log10(
            magnifications[valid_mask] / np.abs(mu_ave)
        )
        calculated_mags = magmap_instance.magnitudes
        assert calculated_mags.shape == expected_mags.shape
        np.testing.assert_allclose(
            calculated_mags[valid_mask], expected_mags[valid_mask], rtol=1e-6
        )
        assert np.all(
            np.isnan(calculated_mags[~valid_mask])
            | np.isinf(calculated_mags[~valid_mask])
        )

    # Test methods
    def test_get_pixel_size_meters(self, magmap_instance, cosmology):
        """Tests calculating pixel size in meters."""
        source_z = 1.5
        pix_size_meters = magmap_instance.get_pixel_size_meters(source_z, cosmology)
        assert isinstance(pix_size_meters, float)
        assert pix_size_meters > 0
        pixel_size_arcsec = magmap_instance.pixel_size
        ang_diam_dist_m = cosmology.angular_diameter_distance(source_z).to(u.m).value
        expected_meters = ang_diam_dist_m * pixel_size_arcsec * u.arcsec.to(u.rad)
        np.testing.assert_allclose(pix_size_meters, expected_meters)
