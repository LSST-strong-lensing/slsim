import pytest
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from unittest.mock import patch  # Keep patch for testing caching logic

# Import the classes to be tested
from slsim.Microlensing.source_morphology.source_morphology import (
    SourceMorphology,
)
from slsim.Microlensing.source_morphology.gaussian import (
    GaussianSourceMorphology,
)
from slsim.Microlensing.source_morphology.agn import AGNSourceMorphology
from slsim.Microlensing.source_morphology.supernovae import (
    SupernovaeSourceMorphology,
)

# Import the real dependencies (make sure they are installed/available)
try:
    import speclite.filters

    SPECLITE_AVAILABLE = True
except ImportError:
    SPECLITE_AVAILABLE = False

try:
    from slsim.Util.astro_util import (
        calculate_accretion_disk_emission,
        calculate_gravitational_radius,
    )

    ASTRO_UTIL_AVAILABLE = True
except ImportError:
    ASTRO_UTIL_AVAILABLE = False

    # Define dummy functions if unavailable, so tests requiring them can be skipped
    def calculate_gravitational_radius(*args, **kwargs):
        raise ImportError("slsim.Util.astro_util not found")

    def calculate_accretion_disk_emission(*args, **kwargs):
        raise ImportError("slsim.Util.astro_util not found")


# Test Fixtures


@pytest.fixture
def cosmology():
    """Provides a cosmology instance for testing."""
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def kwargs_Gaussian(cosmology):
    """Provides a dictionary of parameters for the Gaussian source."""
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "source_size": 0.05,  # FWHM in arcsec
        "length_x": 1,  # in arcsec
        "length_y": 1,  # in arcsec
        "num_pix_x": 100,
        "num_pix_y": 100,
    }


@pytest.fixture
def kwargs_Gaussian_centered(cosmology):
    """Provides a dictionary of parameters for a centered Gaussian source."""
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "source_size": 0.05,
        "length_x": 1,
        "length_y": 1,
        "num_pix_x": 101,  # Odd number for clear center
        "num_pix_y": 101,
        "center_x": 0.1,  # Test non-zero center
        "center_y": -0.1,
    }


@pytest.fixture
def kwargs_AGN_band(cosmology):
    """Provides a dictionary of parameters for the AGN source using a band."""
    # r_resolution might influence sampling density, not necessarily output grid size
    resolution = 100
    return {
        "source_redshift": 0.1,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": resolution,
        "black_hole_mass_exponent": 8.0,
        "inclination_angle": 0,
        "black_hole_spin": 0,
        "observing_wavelength_band": "r",  # Requires speclite
        "eddington_ratio": 0.15,
    }


@pytest.fixture
def kwargs_AGN_wave(cosmology):
    """Provides a dictionary of parameters for the AGN source using
    wavelength."""
    resolution = 100
    return {
        "source_redshift": 0.1,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": resolution,
        "black_hole_mass_exponent": 8.0,
        "inclination_angle": 30,  # Test non-zero inclination
        "black_hole_spin": 0.5,  # Test non-zero spin
        "observer_frame_wavelength_in_nm": 600,
        "eddington_ratio": 0.15,
    }


# --- Fixtures for Source Instances ---


@pytest.fixture
def gaussian_source(kwargs_Gaussian):
    """Provides an instance of GaussianSourceMorphology."""
    return GaussianSourceMorphology(**kwargs_Gaussian)


@pytest.fixture
def gaussian_source_centered(kwargs_Gaussian_centered):
    """Provides an instance of centered GaussianSourceMorphology."""
    return GaussianSourceMorphology(**kwargs_Gaussian_centered)


# Fixtures that require external dependencies
@pytest.fixture()
def agn_source_band(kwargs_AGN_band):
    """Provides an instance of AGNSourceMorphology using band."""
    if not ASTRO_UTIL_AVAILABLE or not SPECLITE_AVAILABLE:
        pytest.skip("Requires slsim.Util.astro_util and speclite")
    return AGNSourceMorphology(**kwargs_AGN_band)


@pytest.fixture()
def agn_source_wave(kwargs_AGN_wave):
    """Provides an instance of AGNSourceMorphology using wavelength."""
    if not ASTRO_UTIL_AVAILABLE:
        pytest.skip("Requires slsim.Util.astro_util")
    return AGNSourceMorphology(**kwargs_AGN_wave)


# ---- Test Class ----


class TestSourceMorphology:
    """Tests the SourceMorphology base class methods indirectly and
    directly."""

    def test_base_class_init(self):
        base_source = SourceMorphology()
        assert isinstance(base_source, SourceMorphology)
        assert not hasattr(base_source, "_kernel_map")
        assert not hasattr(base_source, "_length_x")

    def test_base_class_get_kernel_map_not_implemented(self):
        base_source = SourceMorphology()
        with pytest.raises(NotImplementedError):
            base_source.get_kernel_map()
        with pytest.raises(NotImplementedError):
            _ = base_source.kernel_map

    def test_base_class_pixel_scale_attribute_error(self):
        base_source = SourceMorphology()
        with pytest.raises(AttributeError):
            _ = base_source.length_x
        with pytest.raises(AttributeError):
            _ = base_source.pixel_scale_x
        with pytest.raises(AttributeError, match="Pixel scale not defined."):
            _ = base_source.pixel_scale
        with pytest.raises(AttributeError):
            _ = base_source.pixel_scale_x_m
        with pytest.raises(AttributeError):
            _ = base_source.pixel_scale_y_m
        with pytest.raises(AttributeError):
            _ = base_source.pixel_scale_m

    def test_conversions(self, gaussian_source, cosmology):
        arcsecs_val = 1.0
        redshift = gaussian_source.source_redshift
        metres_val = gaussian_source.arcsecs_to_metres(arcsecs_val, cosmology, redshift)
        assert isinstance(metres_val, float) and metres_val > 0
        arcsecs_val_round_trip = gaussian_source.metres_to_arcsecs(
            metres_val, cosmology, redshift
        )
        np.testing.assert_allclose(arcsecs_val_round_trip, arcsecs_val, rtol=1e-6)
        assert gaussian_source.arcsecs_to_metres(0, cosmology, redshift) == 0.0
        assert gaussian_source.metres_to_arcsecs(0, cosmology, redshift) == 0.0

    # Keep patch here to test the caching mechanism itself
    def test_property_caching(self, gaussian_source, agn_source_wave):  # Use fixture
        """Tests that properties like kernel_map and derived scales are
        cached."""
        # --- Test Gaussian kernel_map caching ---
        # assert hasattr(gaussian_source, "_kernel_map")
        initial_kernel = gaussian_source.kernel_map.copy()
        with patch.object(
            GaussianSourceMorphology,
            "get_kernel_map",
            wraps=gaussian_source.get_kernel_map,
        ) as mock_get_kernel:
            del gaussian_source._kernel_map
            kernel1 = gaussian_source.kernel_map
            mock_get_kernel.assert_called_once()
            kernel2 = gaussian_source.kernel_map
            mock_get_kernel.assert_called_once()
            assert np.array_equal(kernel1, kernel2)
            assert hasattr(gaussian_source, "_kernel_map")
            assert np.array_equal(kernel1, initial_kernel)

        # --- Test Gaussian pixel_scale caching ---
        assert hasattr(gaussian_source, "_pixel_scale")
        initial_pixel_scale = gaussian_source._pixel_scale
        del gaussian_source._pixel_scale
        scale1 = gaussian_source.pixel_scale
        assert hasattr(gaussian_source, "_pixel_scale")
        assert scale1 == initial_pixel_scale
        scale2 = gaussian_source.pixel_scale
        assert scale1 == scale2

        # --- Test AGN meter scale caching ---
        # Use the agn_source_wave fixture instance provided by pytest
        local_agn_source = agn_source_wave
        assert hasattr(local_agn_source, "_pixel_scale_x_m")
        assert hasattr(local_agn_source, "_pixel_scale_y_m")
        assert hasattr(local_agn_source, "_pixel_scale_m")
        assert hasattr(local_agn_source, "_pixel_scale")

        with patch.object(
            AGNSourceMorphology,
            "arcsecs_to_metres",
            wraps=local_agn_source.arcsecs_to_metres,
        ) as mock_arcsecs_to_metres:
            if hasattr(local_agn_source, "_pixel_scale_x_m"):
                del local_agn_source._pixel_scale_x_m
            if hasattr(local_agn_source, "_pixel_scale_y_m"):
                del local_agn_source._pixel_scale_y_m
            if hasattr(local_agn_source, "_pixel_scale_m"):
                del local_agn_source._pixel_scale_m

            _ = local_agn_source.pixel_scale_x_m  # -> Calls arcsecs_to_metres 1st time
            assert mock_arcsecs_to_metres.call_count == 1
            _ = local_agn_source.pixel_scale_y_m  # -> Calls arcsecs_to_metres 2nd time
            assert mock_arcsecs_to_metres.call_count == 2
            _ = local_agn_source.pixel_scale_m  # -> Calls arcsecs_to_metres 3rd time
            assert mock_arcsecs_to_metres.call_count == 3

            # Access again, should use caches
            _ = local_agn_source.pixel_scale_x_m
            assert mock_arcsecs_to_metres.call_count == 3
            _ = local_agn_source.pixel_scale_y_m
            assert mock_arcsecs_to_metres.call_count == 3
            _ = local_agn_source.pixel_scale_m
            assert mock_arcsecs_to_metres.call_count == 3
            assert hasattr(local_agn_source, "_pixel_scale_x_m")
            assert hasattr(local_agn_source, "_pixel_scale_y_m")
            assert hasattr(local_agn_source, "_pixel_scale_m")


class TestGaussianSourceMorphology:
    """Tests the GaussianSourceMorphology class."""

    def test_initialization(self, gaussian_source, kwargs_Gaussian):
        assert gaussian_source.source_redshift == kwargs_Gaussian["source_redshift"]
        assert gaussian_source.cosmo == kwargs_Gaussian["cosmo"]
        assert gaussian_source.source_size == kwargs_Gaussian["source_size"]
        assert gaussian_source.length_x == kwargs_Gaussian["length_x"]
        assert gaussian_source.length_y == kwargs_Gaussian["length_y"]
        assert gaussian_source.num_pix_x == kwargs_Gaussian["num_pix_x"]
        assert gaussian_source.num_pix_y == kwargs_Gaussian["num_pix_y"]
        assert gaussian_source.center_x == 0
        assert gaussian_source.center_y == 0

        expected_pix_scale_x = (
            kwargs_Gaussian["length_x"] / kwargs_Gaussian["num_pix_x"]
        )
        expected_pix_scale_y = (
            kwargs_Gaussian["length_y"] / kwargs_Gaussian["num_pix_y"]
        )
        expected_pix_scale = np.sqrt(expected_pix_scale_x * expected_pix_scale_y)

        assert gaussian_source.pixel_scale_x == expected_pix_scale_x
        assert gaussian_source.pixel_scale_y == expected_pix_scale_y
        assert gaussian_source.pixel_scale == expected_pix_scale
        # assert hasattr(gaussian_source, "_kernel_map")
        assert isinstance(gaussian_source.kernel_map, np.ndarray)

    def test_get_kernel_map(self, gaussian_source, kwargs_Gaussian):
        kernel = gaussian_source.get_kernel_map()
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (
            kwargs_Gaussian["num_pix_y"],
            kwargs_Gaussian["num_pix_x"],
        )
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-6)
        center_iy = kwargs_Gaussian["num_pix_y"] // 2
        center_ix = kwargs_Gaussian["num_pix_x"] // 2
        peak_indices = np.unravel_index(np.argmax(kernel), kernel.shape)
        assert abs(peak_indices[0] - center_iy) <= 1
        assert abs(peak_indices[1] - center_ix) <= 1
        assert np.all(kernel >= 0)

    def test_get_kernel_map_centered(
        self, gaussian_source_centered, kwargs_Gaussian_centered
    ):
        kernel = gaussian_source_centered.kernel_map
        num_pix_x = kwargs_Gaussian_centered["num_pix_x"]
        num_pix_y = kwargs_Gaussian_centered["num_pix_y"]
        length_x = kwargs_Gaussian_centered["length_x"]
        length_y = kwargs_Gaussian_centered["length_y"]
        center_x = kwargs_Gaussian_centered["center_x"]
        center_y = kwargs_Gaussian_centered["center_y"]

        assert kernel.shape == (num_pix_y, num_pix_x)
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-6)

        xs = np.linspace(center_x - length_x / 2, center_x + length_x / 2, num_pix_x)
        ys = np.linspace(center_y - length_y / 2, center_y + length_y / 2, num_pix_y)
        expected_ix = np.argmin(np.abs(xs - center_x))
        expected_iy = np.argmin(np.abs(ys - center_y))
        peak_indices = np.unravel_index(np.argmax(kernel), kernel.shape)
        assert peak_indices[0] == expected_iy
        assert peak_indices[1] == expected_ix

    def test_properties(self, gaussian_source, kwargs_Gaussian):
        assert gaussian_source.length_x == kwargs_Gaussian["length_x"]
        assert gaussian_source.length_y == kwargs_Gaussian["length_y"]
        assert gaussian_source.num_pix_x == kwargs_Gaussian["num_pix_x"]
        assert gaussian_source.num_pix_y == kwargs_Gaussian["num_pix_y"]
        assert gaussian_source.pixel_scale_x == gaussian_source._pixel_scale_x
        assert gaussian_source.pixel_scale_y == gaussian_source._pixel_scale_y
        assert gaussian_source.pixel_scale == gaussian_source._pixel_scale

        pix_scale_x_m = gaussian_source.pixel_scale_x_m
        pix_scale_y_m = gaussian_source.pixel_scale_y_m
        pix_scale_m = gaussian_source.pixel_scale_m

        assert isinstance(pix_scale_x_m, float) and pix_scale_x_m >= 0
        assert isinstance(pix_scale_y_m, float) and pix_scale_y_m >= 0
        assert isinstance(pix_scale_m, float) and pix_scale_m >= 0
        if pix_scale_x_m > 0 and pix_scale_y_m > 0:
            np.testing.assert_allclose(
                pix_scale_m, np.sqrt(pix_scale_x_m * pix_scale_y_m), rtol=1e-6
            )

        expected_pix_scale_x_m = gaussian_source.arcsecs_to_metres(
            gaussian_source.pixel_scale_x,
            gaussian_source.cosmo,
            gaussian_source.source_redshift,
        )
        np.testing.assert_allclose(pix_scale_x_m, expected_pix_scale_x_m, rtol=1e-6)


# Apply skips at the class level
@pytest.mark.skipif(not ASTRO_UTIL_AVAILABLE, reason="slsim.Util.astro_util not found")
class TestAGNSourceMorphology:
    """Tests the AGNSourceMorphology class using real dependencies."""

    # Apply specific skip for speclite dependency here
    @pytest.mark.skipif(not SPECLITE_AVAILABLE, reason="speclite not found")
    def test_initialization_band(self, agn_source_band, kwargs_AGN_band):
        """Tests AGN initialization using a wavelength band."""
        assert agn_source_band.source_redshift == kwargs_AGN_band["source_redshift"]
        assert (
            agn_source_band.observing_wavelength_band
            == kwargs_AGN_band["observing_wavelength_band"]
        )
        assert (
            agn_source_band.black_hole_mass
            == 10 ** kwargs_AGN_band["black_hole_mass_exponent"]
        )

        filter_r = speclite.filters.load_filter("lsst2023-r")
        expected_wave_nm = filter_r.effective_wavelength.to(u.nm).value
        assert hasattr(agn_source_band, "observer_frame_wavelength_in_nm")
        np.testing.assert_allclose(
            agn_source_band.observer_frame_wavelength_in_nm, expected_wave_nm
        )

        # assert hasattr(agn_source_band, "_kernel_map")
        kernel = agn_source_band.kernel_map
        assert isinstance(kernel, np.ndarray)
        # Assert actual shape, don't assume based on r_resolution
        assert (
            kernel.shape[0] > 0 and kernel.shape[1] > 0
        )  # Basic check for valid shape

        # Check scales were calculated
        assert (
            hasattr(agn_source_band, "_pixel_scale_x_m")
            and agn_source_band.pixel_scale_x_m > 0
        )
        assert (
            hasattr(agn_source_band, "_pixel_scale_y_m")
            and agn_source_band.pixel_scale_y_m > 0
        )
        assert (
            hasattr(agn_source_band, "_pixel_scale_m")
            and agn_source_band.pixel_scale_m > 0
        )
        assert (
            hasattr(agn_source_band, "_pixel_scale_x")
            and agn_source_band.pixel_scale_x > 0
        )
        assert (
            hasattr(agn_source_band, "_pixel_scale_y")
            and agn_source_band.pixel_scale_y > 0
        )
        assert (
            hasattr(agn_source_band, "_pixel_scale") and agn_source_band.pixel_scale > 0
        )
        assert hasattr(agn_source_band, "_length_x") and agn_source_band.length_x > 0
        assert hasattr(agn_source_band, "_length_y") and agn_source_band.length_y > 0
        # Check num_pix property matches actual kernel shape
        assert agn_source_band.num_pix_x == kernel.shape[1]  # x = columns
        assert agn_source_band.num_pix_y == kernel.shape[0]  # y = rows

    def test_initialization_wavelength(self, agn_source_wave, kwargs_AGN_wave):
        """Tests AGN initialization using a specific wavelength."""
        assert agn_source_wave.source_redshift == kwargs_AGN_wave["source_redshift"]
        assert (
            agn_source_wave.observer_frame_wavelength_in_nm
            == kwargs_AGN_wave["observer_frame_wavelength_in_nm"]
        )
        assert agn_source_wave.observing_wavelength_band is None
        assert agn_source_wave.inclination_angle == kwargs_AGN_wave["inclination_angle"]
        assert agn_source_wave.black_hole_spin == kwargs_AGN_wave["black_hole_spin"]

        # assert hasattr(agn_source_wave, "_kernel_map")
        kernel = agn_source_wave.kernel_map
        assert isinstance(kernel, np.ndarray)
        # Assert actual shape
        assert kernel.shape[0] > 0 and kernel.shape[1] > 0

        assert (
            hasattr(agn_source_wave, "_pixel_scale_x_m")
            and agn_source_wave.pixel_scale_x_m > 0
        )
        assert (
            isinstance(agn_source_wave.length_x, float) and agn_source_wave.length_x > 0
        )
        # Check num_pix property matches actual kernel shape
        assert agn_source_wave.num_pix_x == kernel.shape[1]  # x = columns
        assert agn_source_wave.num_pix_y == kernel.shape[0]  # y = rows

    def test_get_kernel_map(self, agn_source_wave, kwargs_AGN_wave):
        """Tests the generated AGN kernel map properties."""
        kernel_prop = agn_source_wave.kernel_map
        kernel_direct = agn_source_wave.get_kernel_map()

        assert isinstance(kernel_direct, np.ndarray)
        # Assert actual shape
        actual_shape = kernel_direct.shape
        assert actual_shape[0] > 0 and actual_shape[1] > 0

        np.testing.assert_allclose(np.sum(kernel_direct), 1.0, rtol=1e-5)
        assert np.all(kernel_direct >= 0)
        assert np.array_equal(kernel_prop, kernel_direct)

    def test_properties(self, agn_source_wave, kwargs_AGN_wave):
        """Tests the property accessors of the AGN source and scale
        consistency."""
        # Get actual kernel shape from the created object
        kernel_shape = agn_source_wave.kernel_map.shape  # (rows, cols) = (y, x)

        # Calculate expected scales based on real grav radius and ACTUAL kernel shape
        grav_radius = calculate_gravitational_radius(
            kwargs_AGN_wave["black_hole_mass_exponent"]
        )
        grav_radius_val = grav_radius.to(u.m).value

        # Use correct axes: x scale depends on cols (shape[1]), y scale depends on rows (shape[0])
        expected_pix_scale_x_m = (
            2 * kwargs_AGN_wave["r_out"] * grav_radius_val / kernel_shape[1]
        )
        expected_pix_scale_y_m = (
            2 * kwargs_AGN_wave["r_out"] * grav_radius_val / kernel_shape[0]
        )
        expected_pix_scale_m = np.sqrt(expected_pix_scale_x_m * expected_pix_scale_y_m)

        # Test calculated meter scales using property access
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_x_m, expected_pix_scale_x_m, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_y_m, expected_pix_scale_y_m, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_m, expected_pix_scale_m, rtol=1e-6
        )

        # Test conversion back to arcsec using property access
        expected_pix_scale_x = agn_source_wave.metres_to_arcsecs(
            expected_pix_scale_x_m,
            agn_source_wave.cosmo,
            agn_source_wave.source_redshift,
        )
        expected_pix_scale_y = agn_source_wave.metres_to_arcsecs(
            expected_pix_scale_y_m,
            agn_source_wave.cosmo,
            agn_source_wave.source_redshift,
        )
        expected_pix_scale = np.sqrt(expected_pix_scale_x * expected_pix_scale_y)

        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_x, expected_pix_scale_x, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_y, expected_pix_scale_y, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale, expected_pix_scale, rtol=1e-6
        )

        # Test num pix and length using property access
        expected_num_pix_x = kernel_shape[1]
        expected_num_pix_y = kernel_shape[0]
        expected_length_x = expected_num_pix_x * expected_pix_scale_x
        expected_length_y = expected_num_pix_y * expected_pix_scale_y

        assert agn_source_wave.num_pix_x == expected_num_pix_x
        assert agn_source_wave.num_pix_y == expected_num_pix_y
        np.testing.assert_allclose(
            agn_source_wave.length_x, expected_length_x, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.length_y, expected_length_y, rtol=1e-6
        )

    def test_agn_not_implemented_methods(self, agn_source_wave):
        """Tests that specific AGN methods raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="This method is not implemented yet."
        ):
            agn_source_wave.get_variable_kernel_map()
        with pytest.raises(
            NotImplementedError, match="This method is not implemented yet."
        ):
            agn_source_wave.get_integrated_kernel_map(band="r")  # Pass dummy band


class TestSupernovaeSourceMorphology:
    """Tests the SupernovaeSourceMorphology class."""

    def test_init_not_implemented(self):
        """Tests that initializing SupernovaeSourceMorphology raises
        NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Supernovae source morphology is not implemented yet.",
        ):
            SupernovaeSourceMorphology()  # No args needed as error is immediate
