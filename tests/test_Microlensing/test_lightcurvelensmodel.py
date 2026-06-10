import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from unittest.mock import patch

from slsim.Microlensing.lightcurvelensmodel import MicrolensingLightCurveFromLensModel
from slsim.Microlensing.magmap import MagnificationMap

# ==========================================
# 1. CORE FIXTURES & DATA LOADING
# ==========================================


@pytest.fixture(scope="module")
def cosmology():
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture(scope="module")
def lens_source_info():
    return {
        "deflector_redshift": 1.19,
        "source_redshift": 3.40,
        "deflector_velocity_dispersion": 297.6,
        "ra_lens": 10.0,
        "dec_lens": -10.0,
        "theta_star": 1.45e-06,
    }


@pytest.fixture(scope="module")
def microlensing_params():
    """Provides the core microlensing parameters for 4 simulated images."""
    return {
        "kappa_star": np.array([0.12, 0.13, 0.15, 0.21]),
        "kappa_tot": np.array([0.47, 0.49, 0.53, 0.61]),
        "shear": np.array([0.42, 0.46, 0.51, 0.58]),
        "shear_phi": np.array([2.0, -1.8, -0.7, 0.03]),
    }


@pytest.fixture
def base_init_kwargs(lens_source_info, microlensing_params, cosmology):
    """Provides the standard dictionary to initialize
    MicrolensingLightCurveFromLensModel."""
    theta_star = lens_source_info["theta_star"]
    return {
        "source_redshift": lens_source_info["source_redshift"],
        "deflector_redshift": lens_source_info["deflector_redshift"],
        "kappa_star_images": microlensing_params["kappa_star"],
        "kappa_tot_images": microlensing_params["kappa_tot"],
        "shear_images": microlensing_params["shear"],
        "shear_phi_angle_images": microlensing_params["shear_phi"],
        "ra_lens": lens_source_info["ra_lens"],
        "dec_lens": lens_source_info["dec_lens"],
        "deflector_velocity_dispersion": lens_source_info[
            "deflector_velocity_dispersion"
        ],
        "cosmology": cosmology,
        "kwargs_magnification_map": {
            "theta_star": theta_star,
            "center_x": 0,
            "center_y": 0,
            "half_length_x": 2.5 * theta_star,
            "half_length_y": 2.5 * theta_star,
            "num_pixels_x": 50,
            "num_pixels_y": 50,
        },
        "point_source_morphology": "gaussian",
        "kwargs_source_morphology": {
            "source_redshift": lens_source_info["source_redshift"],
            "cosmo": cosmology,
            "source_size": 8e-8,
        },
    }


@pytest.fixture(scope="module")
def saved_magmaps(microlensing_params, lens_source_info):
    """Loads actual saved .npy maps from disk and constructs real
    MagnificationMap instances."""
    num_images = len(microlensing_params["kappa_star"])
    loaded_magmaps = []

    test_dir = os.path.dirname(os.path.abspath(__file__))
    saved_map_directory = os.path.join(
        test_dir, "..", "TestData", "test_magmaps_microlensing"
    )

    for i in range(num_images):
        map_filepath = os.path.join(saved_map_directory, f"magmap_{i}.npy")
        if not os.path.exists(map_filepath):
            pytest.fail(f"Required saved map not found: {map_filepath}")

        mag_data = np.load(map_filepath)

        # Build kwargs for this specific map
        theta_star = lens_source_info["theta_star"]
        map_kwargs = {
            "theta_star": theta_star,
            "center_x": 0,
            "center_y": 0,
            "half_length_x": 2.5 * theta_star,
            "half_length_y": 2.5 * theta_star,
            "num_pixels_x": 50,
            "num_pixels_y": 50,
            "kappa_tot": microlensing_params["kappa_tot"][i],
            "shear": microlensing_params["shear"][i],
            "kappa_star": microlensing_params["kappa_star"][i],
        }
        loaded_magmaps.append(
            MagnificationMap(magnifications_array=mag_data, **map_kwargs)
        )

    return loaded_magmaps


@pytest.fixture
def patched_magmap_generation(saved_magmaps):
    """Patches map generation to return the REAL maps loaded from disk.

    This prevents repetitive @patch decorators on every test.
    """
    with patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
        return_value=saved_magmaps,
    ) as mock:
        yield mock


@pytest.fixture
def safe_ml_model(base_init_kwargs):
    """Returns an initialized model with a safe starting track to avoid map
    boundaries."""
    model = MicrolensingLightCurveFromLensModel(**base_init_kwargs)
    model._lc_start_position = (0, 0)

    num_images = len(model._kappa_star_images)
    model._eff_trv_vel_images = (np.ones(num_images) * 100.0, np.zeros(num_images))
    return model


# ==========================================
# 2. INITIALIZATION & UPDATES
# ==========================================


class TestInitializationAndUpdate:

    def test_missing_morphology_kwargs_raises_error(self, base_init_kwargs):
        args = base_init_kwargs.copy()
        args["point_source_morphology"] = None
        with pytest.raises(
            ValueError, match="point_source_morphology not in kwargs_microlensing"
        ):
            MicrolensingLightCurveFromLensModel(**args)

        args["point_source_morphology"] = "gaussian"
        args["kwargs_source_morphology"] = None
        with pytest.raises(
            ValueError, match="kwargs_source_morphology not in kwargs_microlensing"
        ):
            MicrolensingLightCurveFromLensModel(**args)

    def test_invalid_morphology_type_raises_error(self, base_init_kwargs):
        args = base_init_kwargs.copy()
        args["point_source_morphology"] = "invalid_morph"
        with pytest.raises(ValueError, match="Invalid source morphology type"):
            MicrolensingLightCurveFromLensModel(**args)

    def test_default_magnification_map_kwargs_generation(self, base_init_kwargs):
        args = base_init_kwargs.copy()
        args["kwargs_magnification_map"] = None
        model = MicrolensingLightCurveFromLensModel(**args)

        assert model._kwargs_magnification_map is not None
        assert model._kwargs_magnification_map["num_pixels_x"] == 1000

    def test_morphology_instance_is_pre_instantiated(self, base_init_kwargs):
        from slsim.Microlensing.source_morphology.gaussian import (
            GaussianSourceMorphology,
        )

        model = MicrolensingLightCurveFromLensModel(**base_init_kwargs)

        assert model._source_morphology_instance is not None
        assert isinstance(model._source_morphology_instance, GaussianSourceMorphology)

    def test_update_source_morphology(self, safe_ml_model, base_init_kwargs):
        new_kwargs = base_init_kwargs["kwargs_source_morphology"].copy()
        new_kwargs["source_size"] = 1.0e-7

        safe_ml_model.update_source_morphology(new_kwargs)
        assert safe_ml_model._kwargs_source_morphology["source_size"] == 1.0e-7


# ==========================================
# 3. KINEMATICS & TRACKING TESTS
# ==========================================


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in divide:RuntimeWarning"
)
class TestKinematicsAndTracking:

    def test_effective_velocity_returns_correct_shapes(
        self, safe_ml_model, microlensing_params
    ):
        num_images = len(microlensing_params["kappa_star"])
        velocities, angles = safe_ml_model._effective_transverse_velocity_images(
            random_seed=42
        )

        assert velocities.shape == (num_images,)
        assert angles.shape == (num_images,)
        assert np.all(velocities >= 0)

    def test_effective_velocity_non_magmap_frame(
        self, safe_ml_model, microlensing_params
    ):
        num_images = len(microlensing_params["kappa_star"])
        velocities, angles = safe_ml_model._effective_transverse_velocity_images(
            random_seed=42, magmap_reference_frame=False
        )
        assert angles.shape == (num_images,)

    def test_astropy_quantity_handling_in_velocities(
        self, base_init_kwargs, microlensing_params
    ):
        args = base_init_kwargs.copy()
        args["ra_lens"] *= u.deg
        args["dec_lens"] *= u.deg
        args["deflector_velocity_dispersion"] *= u.km / u.s

        model = MicrolensingLightCurveFromLensModel(**args)
        velocities, _ = model._effective_transverse_velocity_images(random_seed=42)
        assert velocities.shape == (len(microlensing_params["kappa_star"]),)

    def test_effective_velocity_pole_branch(self, base_init_kwargs):
        """Forces the cross-product to hit the np.allclose(e1, 0) branch by
        putting the lens at DEC=90."""
        args = base_init_kwargs.copy()
        args["ra_lens"] = 0 * u.deg
        args["dec_lens"] = 90 * u.deg
        model = MicrolensingLightCurveFromLensModel(**args)
        velocities, _ = model._effective_transverse_velocity_images(random_seed=42)
        assert isinstance(velocities, np.ndarray)

    def test_effective_velocity_property_caching(self, base_init_kwargs):
        """Ensures the property caches the velocities successfully.

        We use a clean model here because safe_ml_model manually
        overwrites the cache.
        """
        model = MicrolensingLightCurveFromLensModel(**base_init_kwargs)
        assert not hasattr(model, "_eff_trv_vel_images")

        res1 = model.effective_transverse_velocity_images
        res2 = model.effective_transverse_velocity_images
        assert res1 is res2

    def test_start_position_caching_and_bounds(self, safe_ml_model):
        del safe_ml_model._lc_start_position  # Remove safe override

        pos1 = safe_ml_model.lc_start_position
        pos2 = safe_ml_model.lc_start_position
        assert pos1 == pos2  # Check caching

        half_x = safe_ml_model._kwargs_magnification_map["half_length_x"]
        assert -half_x <= pos1[0] <= half_x

    def test_explicit_start_position_reset(self, safe_ml_model):
        new_pos = safe_ml_model.reset_start_position(
            x_start_position=1e-6, y_start_position=2e-6
        )
        assert new_pos == (1e-6, 2e-6)
        assert safe_ml_model.lc_start_position == (1e-6, 2e-6)

    def test_random_start_position_reset(self, safe_ml_model):
        """Calling reset without values should randomly generate a valid
        start."""
        res = safe_ml_model.reset_start_position()
        half_x = safe_ml_model._kwargs_magnification_map["half_length_x"]
        assert -half_x <= res[0] <= half_x


# ==========================================
# 4. LIGHTCURVE GENERATION TESTS
# ==========================================


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in divide:RuntimeWarning"
)
class TestLightcurveGeneration:

    def test_interpolate_light_curve(self, safe_ml_model):
        time_orig = np.array([0.0, 10.0, 20.0, 30.0])
        lc_orig = np.array([1.0, 1.5, 1.2, 1.8])
        time_new = np.array([5.0, 15.0, 25.0])

        lc_interp = safe_ml_model._interpolate_light_curve(lc_orig, time_orig, time_new)
        np.testing.assert_allclose(lc_interp, np.array([1.25, 1.35, 1.5]))

    def test_magmaps_images_property_raises_before_generation(self, base_init_kwargs):
        model = MicrolensingLightCurveFromLensModel(**base_init_kwargs)
        with pytest.raises(AttributeError, match="Magnification maps are not set"):
            _ = model.magmaps_images

    @patch("slsim.Microlensing.lightcurvelensmodel.MagnificationMap")
    def test_magmaps_images_caching(
        self, mock_magmap, base_init_kwargs, microlensing_params
    ):
        """Tests that generating magmaps is properly cached by running the real
        method and intercepting the underlying MagnificationMap class
        initialization."""
        model = MicrolensingLightCurveFromLensModel(**base_init_kwargs)

        maps1 = model.generate_magnification_maps_from_microlensing_params()
        maps2 = model.generate_magnification_maps_from_microlensing_params()

        assert maps1 is maps2
        # The MagnificationMap constructor should be called exactly once per image
        assert mock_magmap.call_count == len(microlensing_params["kappa_star"])
        assert model.magmaps_images is maps1

    def test_generate_point_source_lightcurves_structure(
        self, patched_magmap_generation, safe_ml_model, microlensing_params
    ):
        num_images = len(microlensing_params["kappa_star"])
        time_array = np.linspace(0, 4000, 50)

        lightcurves, tracks, time_arrays = (
            safe_ml_model.generate_point_source_lightcurves(
                time_array, lightcurve_type="magnitude", num_lightcurves=1
            )
        )

        patched_magmap_generation.assert_called_once()
        assert len(lightcurves) == num_images
        assert len(lightcurves[0]) == 1  # num_lightcurves
        assert lightcurves[0][0].shape == time_array.shape

    def test_generate_point_source_lightcurves_single_element_time(
        self, patched_magmap_generation, safe_ml_model
    ):
        """Tests the branch where the length of the time array is exactly 1."""
        time_single = np.array([100.0])
        lightcurves, _, _ = safe_ml_model.generate_point_source_lightcurves(time_single)
        assert len(lightcurves[0][0]) == 1

    def test_generate_point_source_lightcurves_2d_time_array(
        self, patched_magmap_generation, safe_ml_model, microlensing_params
    ):
        num_images = len(microlensing_params["kappa_star"])
        time_2d = np.tile(np.linspace(0, 4000, 50), (num_images, 1))

        lightcurves, _, _ = safe_ml_model.generate_point_source_lightcurves(
            time_2d, "magnitude", 1
        )
        assert lightcurves[0][0].shape == (50,)

    def test_generate_magnitudes_array_returns_correct_shape(
        self, patched_magmap_generation, safe_ml_model, microlensing_params
    ):
        time_array = np.linspace(0, 4000, 50)
        num_images = len(microlensing_params["kappa_star"])

        magnitudes = safe_ml_model.generate_point_source_microlensing_magnitudes(
            time_array
        )

        assert magnitudes.shape == (num_images, len(time_array))
        assert not np.any(np.isnan(magnitudes))

    def test_generate_magnitudes_list_time(
        self, patched_magmap_generation, safe_ml_model, microlensing_params
    ):
        """Tests the branch where time is provided as a Python list."""
        num_images = len(microlensing_params["kappa_star"])
        time_list = np.linspace(0, 4000, 50).tolist()

        magnitudes = safe_ml_model.generate_point_source_microlensing_magnitudes(
            time_list
        )
        assert magnitudes.shape == (num_images, 50)

    def test_generate_magnitudes_scalar_time(
        self, patched_magmap_generation, safe_ml_model, microlensing_params
    ):
        """Tests the branch where time is provided as a single scalar
        (int/float)."""
        num_images = len(microlensing_params["kappa_star"])
        magnitudes = safe_ml_model.generate_point_source_microlensing_magnitudes(500.0)

        assert magnitudes.shape == (num_images,)

    def test_invalid_time_format_raises_error(self, safe_ml_model):
        with pytest.raises(
            ValueError, match="Time array not provided in the correct format"
        ):
            safe_ml_model.generate_point_source_microlensing_magnitudes(time="invalid")

    def test_lightcurves_and_tracks_cached_after_generation(
        self, patched_magmap_generation, safe_ml_model, microlensing_params
    ):
        """Ensures that the @property methods for tracks and lightcurves work
        and raise appropriately."""
        with pytest.raises(AttributeError, match="Lightcurves are not set"):
            _ = safe_ml_model.lightcurves

        with pytest.raises(AttributeError, match="Tracks are not set"):
            _ = safe_ml_model.tracks

        time_array = np.linspace(0, 1000, 50)
        safe_ml_model.generate_point_source_microlensing_magnitudes(time_array)

        # After generation, they should be cached
        assert isinstance(safe_ml_model.lightcurves, np.ndarray)
        assert isinstance(safe_ml_model.tracks, list)
        assert len(safe_ml_model.tracks) == len(microlensing_params["kappa_star"])
