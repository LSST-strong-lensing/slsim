import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# Import the class to test
from slsim.Microlensing.lightcurve import (
    MicrolensingLightCurve,
)

# Import supporting classes and functions
from slsim.Microlensing.magmap import MagnificationMap
from slsim.Microlensing.source_morphology.gaussian import GaussianSourceMorphology
from slsim.Microlensing.source_morphology.agn import AGNSourceMorphology

# ---- Test Fixtures ----


@pytest.fixture
def cosmology():
    """Provides a cosmology instance for testing."""
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def theta_star():
    """Provides a theta_star value needed by magmap_instance."""
    return 4e-6  # arcsec


# Create a dummy MagnificationMap class for isolated testing
@pytest.fixture
def magmap_instance(theta_star):  # Request theta_star as argument
    """Provides a basic MagnificationMap instance for testing."""
    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        # Try path relative to test file first
        magmap2D_path = os.path.join(
            test_dir, "..", "TestData", "test_magmaps_microlensing", "magmap_0.npy"
        )

        magmap2D = np.load(magmap2D_path)
    except Exception as e:
        pytest.fail(
            f"Failed to load TestData/test_magmaps_microlensing/magmap_0.npy: {e}"
        )

    # a precomputed map for the parameters below is available in the TestData folder
    # Use the injected theta_star value
    kwargs_MagnificationMap = {
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

    magmap = MagnificationMap(
        magnifications_array=magmap2D,
        **kwargs_MagnificationMap,
    )
    return magmap


@pytest.fixture
def kwargs_source_morphology_Gaussian(cosmology):
    """Provides a Gaussian source morphology kwargs for testing."""
    return {"source_redshift": 0.5, "cosmo": cosmology, "source_size": 1e-7}


@pytest.fixture
def kwargs_source_morphology_AGN_wave(cosmology):
    """Provides an AGN source morphology kwargs (wavelength) for testing."""
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "smbh_mass_exp": 8,
        "inclination_angle": 30,
        "black_hole_spin": 0,
        "observer_frame_wavelength_in_nm": 600,
        "eddington_ratio": 0.1,
    }


@pytest.fixture
def kwargs_source_morphology_AGN_band(cosmology):
    """Provides an AGN source morphology kwargs (band) for testing."""
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "smbh_mass_exp": 8,
        "inclination_angle": 0,
        "black_hole_spin": 0,
        "observing_wavelength_band": "r",
        "eddington_ratio": 0.1,
    }


# --- Fixtures for MicrolensingLightCurve Instances ---


@pytest.fixture
def ml_lc_gaussian(magmap_instance, kwargs_source_morphology_Gaussian):
    return MicrolensingLightCurve(
        magnification_map=magmap_instance,
        time_duration=4000,
        point_source_morphology="gaussian",
        kwargs_source_morphology=kwargs_source_morphology_Gaussian,
    )


@pytest.fixture
def ml_lc_agn_wave(magmap_instance, kwargs_source_morphology_AGN_wave):
    return MicrolensingLightCurve(
        magnification_map=magmap_instance,
        time_duration=4000,
        point_source_morphology="agn",
        kwargs_source_morphology=kwargs_source_morphology_AGN_wave,
    )


@pytest.fixture
def ml_lc_agn_band(magmap_instance, kwargs_source_morphology_AGN_band):
    return MicrolensingLightCurve(
        magnification_map=magmap_instance,
        time_duration=4000,
        point_source_morphology="agn",
        kwargs_source_morphology=kwargs_source_morphology_AGN_band,
    )


# ---- Test Class ----


class TestMicrolensingLightCurve:

    def test_init(self, magmap_instance, kwargs_source_morphology_Gaussian):
        time_dur = 2000
        morphology = "gaussian"
        ml_lc = MicrolensingLightCurve(
            magmap_instance, time_dur, morphology, kwargs_source_morphology_Gaussian
        )
        # Test public properties
        assert ml_lc.magnification_map is magmap_instance
        assert ml_lc.time_duration_observer_frame == time_dur
        # Test internal attributes for those without public properties
        assert ml_lc._point_source_morphology == morphology
        assert ml_lc._kwargs_source_morphology == kwargs_source_morphology_Gaussian
        # Test initialization state
        assert ml_lc._convolved_map is None
        assert ml_lc._source_morphology is None

    def test_properties_access_and_error_handling(self, ml_lc_gaussian):
        """Test the new properties and the error handling for convolved_map."""
        # 1. Test basic properties
        assert isinstance(ml_lc_gaussian.magnification_map, MagnificationMap)
        assert ml_lc_gaussian.time_duration_observer_frame == 4000

        # 2. Test convolved_map property BEFORE generation
        # Should raise ValueError because get_convolved_map() hasn't been called yet
        with pytest.raises(ValueError, match="Convolved map is not initialized"):
            _ = ml_lc_gaussian.convolved_map

        # 3. Generate the map
        ml_lc_gaussian.get_convolved_map()

        # 4. Test convolved_map property AFTER generation
        assert isinstance(ml_lc_gaussian.convolved_map, np.ndarray)

    def test_get_convolved_map_gaussian(self, ml_lc_gaussian, magmap_instance):
        conv_map = ml_lc_gaussian.get_convolved_map(return_source_morphology=False)
        assert isinstance(conv_map, np.ndarray)
        assert conv_map.shape == magmap_instance.magnifications.shape

        # Verify using the new property
        assert ml_lc_gaussian.convolved_map is conv_map
        assert isinstance(ml_lc_gaussian._source_morphology, GaussianSourceMorphology)

        # Reset to test reconstruction
        ml_lc_gaussian._convolved_map = None
        ml_lc_gaussian._source_morphology = None

        conv_map2, morph = ml_lc_gaussian.get_convolved_map(
            return_source_morphology=True
        )
        assert np.array_equal(conv_map, conv_map2)
        assert isinstance(morph, GaussianSourceMorphology)
        assert ml_lc_gaussian._source_morphology is morph

    @pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in divide:RuntimeWarning"
    )
    def test_get_convolved_map_agn_wave(self, ml_lc_agn_wave, magmap_instance):
        conv_map, morph = ml_lc_agn_wave.get_convolved_map(
            return_source_morphology=True
        )
        assert isinstance(conv_map, np.ndarray)
        assert conv_map.shape == magmap_instance.magnifications.shape
        # Verify property access
        assert ml_lc_agn_wave.convolved_map is conv_map
        assert isinstance(morph, AGNSourceMorphology)
        assert ml_lc_agn_wave._source_morphology is morph
        assert hasattr(morph, "pixel_scale_m")

    @pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in divide:RuntimeWarning"
    )
    def test_get_convolved_map_agn_band(self, ml_lc_agn_band, magmap_instance):
        conv_map, morph = ml_lc_agn_band.get_convolved_map(
            return_source_morphology=True
        )
        assert isinstance(conv_map, np.ndarray)
        assert conv_map.shape == magmap_instance.magnifications.shape
        # Verify property access
        assert ml_lc_agn_band.convolved_map is conv_map
        assert isinstance(morph, AGNSourceMorphology)
        assert ml_lc_agn_band._source_morphology is morph
        assert hasattr(morph, "pixel_scale_m")

    def test_get_convolved_map_supernovae(self, ml_lc_gaussian):
        ml_lc_gaussian._point_source_morphology = "supernovae"
        with pytest.raises(NotImplementedError):
            ml_lc_gaussian.get_convolved_map()

    def test_get_convolved_map_invalid_type(self, ml_lc_gaussian):
        ml_lc_gaussian._point_source_morphology = "invalid_type"
        with pytest.raises(ValueError, match="Invalid source morphology type"):
            ml_lc_gaussian.get_convolved_map()

    # --- Tests for generate_lightcurves (using REAL extract_light_curve) ---
    def test_generate_lightcurves_gaussian_magnitude_basic(
        self, ml_lc_gaussian, cosmology
    ):
        """Tests basic light curve generation (Gaussian, magnitude) using real
        extract."""
        num_lc = 3
        try:
            lcs, _tracks, _time_arrays = ml_lc_gaussian.generate_lightcurves(
                source_redshift=0.5,
                cosmo=cosmology,
                lightcurve_type="magnitude",
                num_lightcurves=num_lc,
            )
        except Exception as e:
            pytest.fail(f"generate_lightcurves raised: {e}")
        assert isinstance(lcs, list)
        assert len(lcs) == num_lc
        for lc in lcs:
            assert isinstance(lc, np.ndarray)
            assert len(lc) > 0
            assert np.issubdtype(lc.dtype, np.floating)
            assert not np.any(np.isnan(lc)), "LC has NaNs"
            assert not np.any(np.isinf(lc)), "LC has Infs"

    @pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in divide:RuntimeWarning"
    )
    def test_generate_lightcurves_agn_magnification(self, ml_lc_agn_wave, cosmology):
        """Tests light curve generation (AGN, magnification) using real
        extract."""
        num_lc = 1
        try:
            lcs, _tracks, _time_arrays = ml_lc_agn_wave.generate_lightcurves(
                source_redshift=0.5,
                cosmo=cosmology,
                lightcurve_type="magnification",
                num_lightcurves=num_lc,
            )
        except Exception as e:
            pytest.fail(f"generate_lightcurves raised: {e}")
        assert isinstance(lcs, list)
        assert len(lcs) == num_lc
        lc = lcs[0]
        assert isinstance(lc, np.ndarray)
        assert len(lc) > 0
        assert np.issubdtype(lc.dtype, np.floating)
        assert not np.any(np.isnan(lc)), "LC has NaNs"

    def test_generate_lightcurves_specific_start_and_angle(
        self, ml_lc_gaussian, cosmology
    ):
        """Tests passing specific start position and angle using real
        extract."""
        # coordinates in pixels
        map_shape = ml_lc_gaussian.magnification_map.magnifications.shape
        x_start, y_start, phi = (
            map_shape[1] // 2 + 10,
            map_shape[0] // 2 - 5,
            45.0,
        )  # Offset slightly

        # convert to units of arcseconds
        half_length_x = ml_lc_gaussian.magnification_map.half_length_x
        half_length_y = ml_lc_gaussian.magnification_map.half_length_y
        num_pix_x = ml_lc_gaussian.magnification_map.num_pixels[0]
        num_pix_y = ml_lc_gaussian.magnification_map.num_pixels[1]
        x_start = (x_start - num_pix_x // 2) * 2 * half_length_x / num_pix_x
        y_start = (y_start - num_pix_y // 2) * 2 * half_length_y / num_pix_y

        try:
            lcs, _tracks, _time_arrays = ml_lc_gaussian.generate_lightcurves(
                source_redshift=0.5,
                cosmo=cosmology,
                num_lightcurves=1,
                x_start_position=x_start,
                y_start_position=y_start,
                phi_travel_direction=phi,
            )
            assert isinstance(lcs, list) and len(lcs) == 1
            assert isinstance(lcs[0], np.ndarray) and len(lcs[0]) > 0
        except Exception as e:
            pytest.fail(f"generate_lightcurves raised: {e}")

    def test_generate_lightcurves_invalid_lightcurve_type(
        self, ml_lc_gaussian, cosmology
    ):
        """Tests error handling for invalid lightcurve_type."""
        with pytest.raises(ValueError, match="Lightcurve type not recognized"):
            ml_lc_gaussian.generate_lightcurves(
                0.5, cosmology, lightcurve_type="invalid_one", num_lightcurves=1
            )
