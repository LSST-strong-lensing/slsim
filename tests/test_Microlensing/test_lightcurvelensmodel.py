import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# Import patch and MagicMock for targeted mocking of map generation
from unittest.mock import patch, MagicMock

# Import the class to test
from slsim.Microlensing.lightcurvelensmodel import (
    MicrolensingLightCurveFromLensModel,
)

# Import supporting classes and functions needed for execution
from slsim.Microlensing.magmap import MagnificationMap

# ---- Test Fixtures ----


@pytest.fixture(scope="module")
def microlensing_params():
    """Provides the core microlensing parameters for the images."""
    return {
        "kappa_star": np.array([0.12007537, 0.13209889, 0.15942816, 0.21984733]),
        "kappa_tot": np.array([0.47128266, 0.49348656, 0.53113534, 0.61013069]),
        "shear": np.array([0.42394672, 0.46016948, 0.51043085, 0.58869696]),
        "shear_phi": np.array(
            [2.01471637, -1.81166767, -0.71529481, 0.03913024]
        ),  # degrees
    }


@pytest.fixture(scope="module")
def lens_source_info():
    """Provides redshift, position, and velocity dispersion info."""
    return {
        "deflector_redshift": 1.1901574825480188,
        "source_redshift": 3.406976632521724,
        "deflector_velocity_dispersion": 297.6146094896387,  # km/s
        "ra_lens": 10.0,  # Example RA in degrees
        "dec_lens": -10.0,  # Example DEC in degrees
        "theta_star": 1.4533388875267387e-06,  # arcsec
    }


@pytest.fixture(scope="module")
def cosmology():
    """Provides a cosmology instance for testing."""
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def kwargs_magnification_map_settings(lens_source_info):
    """Provides SETTINGS for MagnificationMap (size, resolution etc)."""
    theta_star = lens_source_info["theta_star"]
    # These SHOULD MATCH THE SAVED MAPS' RESOLUTION AND EXTENT
    num_pix = 50  # Resolution of saved maps
    half_len = 2.5  # Extent (in theta_star units) of saved maps
    return {
        "theta_star": theta_star,
        "center_x": 0,
        "center_y": 0,
        "half_length_x": half_len * theta_star,
        "half_length_y": half_len * theta_star,
        "mass_function": "kroupa",
        "m_solar": 1.0,
        "m_lower": 0.01,
        "m_upper": 5,
        "num_pixels_x": num_pix,
        "num_pixels_y": num_pix,
        "kwargs_IPM": {},  # Added missing key
    }


@pytest.fixture
def kwargs_source_gaussian(lens_source_info, cosmology):
    """Provides keyword arguments for Gaussian source morphology."""
    return {
        "source_redshift": lens_source_info["source_redshift"],
        "cosmo": cosmology,
        "source_size": 8e-8,
    }


@pytest.fixture
def kwargs_source_agn_wave(lens_source_info, cosmology):
    """Provides keyword arguments for AGN (wavelength) source morphology."""
    return {
        "source_redshift": lens_source_info["source_redshift"],
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
def kwargs_source_agn_band(lens_source_info, cosmology):
    """Provides keyword arguments for AGN (band) source morphology."""
    return {
        "source_redshift": lens_source_info["source_redshift"],
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "smbh_mass_exp": 8,
        "inclination_angle": 0,
        "black_hole_spin": 0,
        "observing_wavelength_band": "r",
        "eddington_ratio": 0.1,
    }


@pytest.fixture
def base_init_kwargs(
    microlensing_params,
    lens_source_info,
    cosmology,
    kwargs_magnification_map_settings,
    kwargs_source_gaussian,
):
    """Provides the standard dictionary of arguments used to initialize
    MicrolensingLightCurveFromLensModel.

    Use this to safely create new instances in tests.
    """
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
        "kwargs_magnification_map": kwargs_magnification_map_settings,
        "point_source_morphology": "gaussian",
        "kwargs_source_morphology": kwargs_source_gaussian,
    }


@pytest.fixture
def ml_lens_model(base_init_kwargs):
    """Provides an initialized instance of the class under test using the base
    kwargs."""
    return MicrolensingLightCurveFromLensModel(**base_init_kwargs)


# Helper function to set a safe track for testing to avoid map boundary issues
def add_safe_track(ml_lc_lens_model: MicrolensingLightCurveFromLensModel):
    """Sets a safe start position and velocities to avoid map boundary issues
    during tests."""
    ml_lc_lens_model._lc_start_position = (0, 0)

    num_images = len(ml_lc_lens_model._kappa_star_images)
    safe_velocity = np.ones(num_images) * 100.0  # 100 km/s
    safe_angle = np.zeros(num_images)  # 0 degrees (horizontal)
    ml_lc_lens_model._eff_trv_vel_images = (safe_velocity, safe_angle)


# ---- Helper Function to Load Maps and Create Mock Return Value ---
def create_mock_magmap_list(microlensing_params, kwargs_magnification_map_settings):
    """Loads saved magnification maps and creates MagnificationMap
    instances."""
    num_images = len(microlensing_params["kappa_star"])
    loaded_magmaps = []

    # --- Robust Path Finding ---
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Try relative to test file first (e.g., tests/TestData/...)
    saved_map_directory = os.path.join(
        test_dir, "..", "TestData", "test_magmaps_microlensing"
    )
    # --- End Robust Path Finding ---

    for i in range(num_images):
        map_filename = f"magmap_{i}.npy"
        map_filepath = os.path.join(saved_map_directory, map_filename)

        if not os.path.exists(map_filepath):
            pytest.fail(
                f"Required saved magnification map not found: {map_filepath}. Please check path and filename convention."
            )

        try:
            mag_data = np.load(map_filepath)
        except Exception as e:
            pytest.fail(f"Failed to load map {map_filepath}: {e}")

        current_kwargs = kwargs_magnification_map_settings.copy()
        current_kwargs["kappa_tot"] = microlensing_params["kappa_tot"][i]
        current_kwargs["shear"] = microlensing_params["shear"][i]
        current_kwargs["kappa_star"] = microlensing_params["kappa_star"][i]

        try:
            magmap_obj = MagnificationMap(
                magnifications_array=mag_data, **current_kwargs
            )
            loaded_magmaps.append(magmap_obj)
        except Exception as e:
            pytest.fail(
                f"Failed to instantiate MagnificationMap for image {i} with data from {map_filepath}: {e}"
            )

    return loaded_magmaps


# ---- Test Class ----


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in divide:RuntimeWarning"
)
class TestMicrolensingLightCurveFromLensModel:

    def test_initialization_errors(self, base_init_kwargs):
        """Tests that ValueError is raised for missing morphology kwargs during
        init."""

        # 1. point_source_morphology is None
        with pytest.raises(
            ValueError, match="point_source_morphology not in kwargs_microlensing"
        ):
            args = base_init_kwargs.copy()
            args["point_source_morphology"] = None
            MicrolensingLightCurveFromLensModel(**args)

        # 2. kwargs_source_morphology is None
        with pytest.raises(
            ValueError, match="kwargs_source_morphology not in kwargs_microlensing"
        ):
            args = base_init_kwargs.copy()
            args["kwargs_source_morphology"] = None
            MicrolensingLightCurveFromLensModel(**args)

    def test_initialization_defaults(self, base_init_kwargs):
        """Tests that default kwargs_magnification_map are generated if None
        provided."""
        args = base_init_kwargs.copy()
        args["kwargs_magnification_map"] = None

        # Should not raise ValueError, should print to stdout (captured if -s not used)
        ml_model = MicrolensingLightCurveFromLensModel(**args)

        # Check that defaults were generated
        assert ml_model._kwargs_magnification_map is not None
        assert "theta_star" in ml_model._kwargs_magnification_map
        assert "num_pixels_x" in ml_model._kwargs_magnification_map
        assert ml_model._kwargs_magnification_map["num_pixels_x"] == 1000

    @pytest.mark.parametrize("magmap_frame", [True, False])
    def test_effective_transverse_velocity_images_calculation(
        self,
        ml_lens_model,
        base_init_kwargs,
        microlensing_params,
        magmap_frame,
    ):
        """Test the private method for calculating effective velocities."""
        num_images = len(microlensing_params["shear_phi"])
        velocities, angles = ml_lens_model._effective_transverse_velocity_images(
            random_seed=42,
            magmap_reference_frame=magmap_frame,
        )
        assert isinstance(velocities, np.ndarray)
        assert isinstance(angles, np.ndarray)
        assert velocities.shape == (num_images,)
        assert angles.shape == (num_images,)
        assert np.issubdtype(velocities.dtype, np.floating)
        assert np.issubdtype(angles.dtype, np.floating)
        assert np.all(velocities >= 0)

        # ────── COVER ELSE BRANCHES FOR ra_lens, dec_lens, sig_star AS Quantity ──────

        # Create a copy of args and modify for Quantity inputs
        args_q = base_init_kwargs.copy()
        args_q["ra_lens"] = args_q["ra_lens"] * u.deg
        args_q["dec_lens"] = args_q["dec_lens"] * u.deg
        args_q["deflector_velocity_dispersion"] = (
            args_q["deflector_velocity_dispersion"] * u.km / u.s
        )

        model_q = MicrolensingLightCurveFromLensModel(**args_q)

        velocities_q, angles_q = model_q._effective_transverse_velocity_images(
            random_seed=42, magmap_reference_frame=magmap_frame
        )
        assert velocities_q.shape == (num_images,)
        # ─────────────────────────────────────────────────────────────────────────────

        # ────── COVER the e1 ZERO‐VECTOR BRANCH ──────
        # Choose dec_lens = 90° so u_los is (0,0,1) and first cross yields zero
        args_pole = base_init_kwargs.copy()
        args_pole["ra_lens"] = 0 * u.deg
        args_pole["dec_lens"] = 90 * u.deg

        model_pole = MicrolensingLightCurveFromLensModel(**args_pole)

        v_pole, a_pole = model_pole._effective_transverse_velocity_images(
            random_seed=42, magmap_reference_frame=magmap_frame
        )
        assert isinstance(v_pole, np.ndarray)
        # ─────────────────────────────────────────────────────────────────────────

    def test_effective_transverse_velocity_images_property(self, ml_lens_model):
        """Test the public property and its caching behavior."""

        # Ensure it's not set initially
        assert not hasattr(ml_lens_model, "_eff_trv_vel_images")

        # Access property
        res = ml_lens_model.effective_transverse_velocity_images
        velocities, angles = res

        # Check cache is set
        assert hasattr(ml_lens_model, "_eff_trv_vel_images")
        assert ml_lens_model._eff_trv_vel_images is res

        # Access again, verify identity (caching)
        res_2 = ml_lens_model.effective_transverse_velocity_images
        assert res_2 is res

    def test_lc_start_position(self, ml_lens_model):
        """Test the lightcurve start position property and caching."""

        # Ensure it's not set initially
        assert not hasattr(ml_lens_model, "_lc_start_position")

        # Access property
        pos = ml_lens_model.lc_start_position
        x_start, y_start = pos

        # Check values are within bounds of the map settings
        half_x = ml_lens_model._kwargs_magnification_map["half_length_x"]
        half_y = ml_lens_model._kwargs_magnification_map["half_length_y"]
        assert -half_x <= x_start <= half_x
        assert -half_y <= y_start <= half_y

        # Check caching
        assert hasattr(ml_lens_model, "_lc_start_position")
        pos_2 = ml_lens_model.lc_start_position
        assert pos_2 == pos

    def test_update_source_morphology(self, ml_lens_model):
        """Test the method to update source morphology kwargs."""
        new_morphology = {"test_param": "test_value"}
        ml_lens_model.update_source_morphology(new_morphology)
        assert ml_lens_model._kwargs_source_morphology == new_morphology

    def test_interpolate_light_curve(self, ml_lens_model):
        time_orig = np.array([0.0, 10.0, 20.0, 30.0])
        lc_orig = np.array([1.0, 1.5, 1.2, 1.8])
        time_new = np.array([5.0, 15.0, 25.0])
        lc_interp = ml_lens_model._interpolate_light_curve(lc_orig, time_orig, time_new)
        assert isinstance(lc_interp, np.ndarray)
        assert lc_interp.shape == time_new.shape
        np.testing.assert_allclose(lc_interp, np.array([1.25, 1.35, 1.5]))
        time_new_endpoints = np.array([0.0, 30.0])
        lc_interp_endpoints = ml_lens_model._interpolate_light_curve(
            lc_orig, time_orig, time_new_endpoints
        )
        np.testing.assert_allclose(lc_interp_endpoints, np.array([1.0, 1.8]))

    def test_mocked_generate_magnification_maps(
        self, ml_lens_model, microlensing_params, kwargs_magnification_map_settings
    ):
        """Test magnification map generation with mocking and internal
        storage."""
        num_images = len(microlensing_params["kappa_star"])

        # Create mock maps
        mock_map_list = create_mock_magmap_list(
            microlensing_params, kwargs_magnification_map_settings
        )

        # Test the magmaps_images property before generation
        with pytest.raises(AttributeError, match="Magnification maps are not set"):
            _ = ml_lens_model.magmaps_images

        # check no _magmaps_images set yet
        assert not hasattr(ml_lens_model, "_magmaps_images")

        # Mock the MagnificationMap constructor to avoid GPU computation
        with patch(
            "slsim.Microlensing.lightcurvelensmodel.MagnificationMap"
        ) as mock_magmap_class:
            # Configure the mock to return pre-created mock objects in sequence
            mock_magmap_class.side_effect = mock_map_list

            result = (
                ml_lens_model.generate_magnification_maps_from_microlensing_params()
            )

            # Verify the mock was called correctly
            assert mock_magmap_class.call_count == num_images

            # Verify the method behavior
            assert isinstance(result, list)
            assert len(result) == num_images
            assert result == mock_map_list

            # Verify that _magmaps_images is set correctly
            assert hasattr(ml_lens_model, "_magmaps_images")
            assert ml_lens_model._magmaps_images == mock_map_list

            # Verify that magmaps_images property works
            magmaps_images = ml_lens_model.magmaps_images
            assert len(magmaps_images) == num_images
            assert magmaps_images == mock_map_list

            # Verify individual map properties
            for i, magmap_obj in enumerate(result):
                assert isinstance(magmap_obj, MagnificationMap)
                assert hasattr(magmap_obj, "magnifications")
                assert magmap_obj.magnifications is not None
                assert magmap_obj._kappa_tot == microlensing_params["kappa_tot"][i]
                assert magmap_obj._shear == microlensing_params["shear"][i]
                assert magmap_obj._kappa_star == microlensing_params["kappa_star"][i]
                assert isinstance(magmap_obj, MagnificationMap)

            # Test Caching: Call the method a second time
            result_cached = (
                ml_lens_model.generate_magnification_maps_from_microlensing_params()
            )

            assert result_cached is result

            # Assert that the constructor was NOT called again
            assert mock_magmap_class.call_count == num_images

    @pytest.mark.parametrize(
        "morphology_key, kwargs_source",
        [("gaussian", "kwargs_source_gaussian"), ("agn", "kwargs_source_agn_wave")],
    )
    @pytest.mark.parametrize("lightcurve_type", ["magnitude", "magnification"])
    @pytest.mark.parametrize("num_lc", [1, 3])
    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_lightcurves_structure(
        self,
        mock_generate_maps,
        base_init_kwargs,
        microlensing_params,
        kwargs_magnification_map_settings,
        morphology_key,
        kwargs_source,
        lightcurve_type,
        num_lc,
        request,
    ):
        mock_generate_maps.return_value = create_mock_magmap_list(
            microlensing_params, kwargs_magnification_map_settings
        )
        kwargs_morphology = request.getfixturevalue(kwargs_source)
        num_images = len(microlensing_params["kappa_star"])
        time_array = np.linspace(0, 4000, 100)

        # Create instance for this specific test case by copying base kwargs
        args = base_init_kwargs.copy()
        args["point_source_morphology"] = morphology_key
        args["kwargs_source_morphology"] = kwargs_morphology

        ml_model = MicrolensingLightCurveFromLensModel(**args)
        add_safe_track(ml_model)

        try:
            lightcurves, tracks, time_arrays = (
                ml_model.generate_point_source_lightcurves(
                    time_array,
                    lightcurve_type,
                    num_lc,
                )
            )
        except Exception as e:
            pytest.fail(f"generate_point_source_lightcurves raised: {e}")

        mock_generate_maps.assert_called_once_with()
        assert isinstance(lightcurves, list)
        assert len(lightcurves) == num_images
        assert isinstance(tracks, list)
        assert len(tracks) == num_images
        assert isinstance(time_arrays, list)
        assert len(time_arrays) == num_images
        assert isinstance(lightcurves[0], list)
        assert len(lightcurves[0]) == num_lc
        assert isinstance(tracks[0], list)
        assert len(tracks[0]) == num_lc
        assert isinstance(time_arrays[0], list)
        assert len(time_arrays[0]) == num_lc
        lc00 = lightcurves[0][0]
        track00 = tracks[0][0]
        time00 = time_arrays[0][0]
        assert isinstance(lc00, np.ndarray)
        assert lc00.shape == time_array.shape
        assert np.issubdtype(lc00.dtype, np.floating)
        assert not np.any(np.isnan(lc00)) and not np.any(np.isinf(lc00))
        # --- Modified Assertion for Track ---
        assert isinstance(track00, np.ndarray)
        assert track00.shape[0] == 2
        assert (
            track00.shape[1] > 0
        )  # Check track has some length, not necessarily same as interpolated LC
        # --- End Modified Assertion ---
        assert isinstance(time00, np.ndarray)
        assert time00.shape == time_array.shape
        np.testing.assert_allclose(time00, time_array)

    # Patch map generation even for invalid time test to avoid IPM errors
    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_lightcurves_invalid_time(
        self, mock_generate_maps, ml_lens_model
    ):
        """Tests error handling for invalid time input."""
        mock_generate_maps.return_value = [MagicMock()]
        with pytest.raises(ValueError, match="Time array not provided"):
            ml_lens_model.generate_point_source_lightcurves(time="invalid")

    @pytest.mark.parametrize(
        "morphology_key, kwargs_source",
        [("gaussian", "kwargs_source_gaussian"), ("agn", "kwargs_source_agn_wave")],
    )
    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_microlensing_magnitudes_array_time(
        self,
        mock_generate_maps,
        base_init_kwargs,
        microlensing_params,
        kwargs_magnification_map_settings,
        morphology_key,
        kwargs_source,
        request,
    ):
        mock_generate_maps.return_value = create_mock_magmap_list(
            microlensing_params, kwargs_magnification_map_settings
        )
        kwargs_morphology = request.getfixturevalue(kwargs_source)
        num_images = len(microlensing_params["kappa_star"])
        time_array = np.linspace(0, 4000, 50)

        # Create instance for this specific test case
        args = base_init_kwargs.copy()
        args["point_source_morphology"] = morphology_key
        args["kwargs_source_morphology"] = kwargs_morphology

        ml_model = MicrolensingLightCurveFromLensModel(**args)
        add_safe_track(ml_model)

        try:
            magnitudes = ml_model.generate_point_source_microlensing_magnitudes(
                time_array
            )
        except Exception as e:
            pytest.fail(f"generate_..._magnitudes raised: {e}")

        mock_generate_maps.assert_called_once()
        assert isinstance(magnitudes, np.ndarray)
        assert magnitudes.shape == (num_images, len(time_array))
        assert np.issubdtype(magnitudes.dtype, np.floating)
        assert not np.any(np.isnan(magnitudes)) and not np.any(np.isinf(magnitudes))

    @pytest.mark.parametrize(
        "morphology_key, kwargs_source",
        [("gaussian", "kwargs_source_gaussian"), ("agn", "kwargs_source_agn_wave")],
    )
    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_microlensing_magnitudes_list_time(
        self,
        mock_generate_maps,
        base_init_kwargs,
        microlensing_params,
        kwargs_magnification_map_settings,
        morphology_key,
        kwargs_source,
        request,
    ):
        mock_generate_maps.return_value = create_mock_magmap_list(
            microlensing_params, kwargs_magnification_map_settings
        )
        kwargs_morphology = request.getfixturevalue(kwargs_source)
        num_images = len(microlensing_params["kappa_star"])
        time_array = np.linspace(0, 4000, 50).tolist()  # Convert to list

        args = base_init_kwargs.copy()
        args["point_source_morphology"] = morphology_key
        args["kwargs_source_morphology"] = kwargs_morphology

        ml_model = MicrolensingLightCurveFromLensModel(**args)
        add_safe_track(ml_model)

        try:
            magnitudes = ml_model.generate_point_source_microlensing_magnitudes(
                time_array
            )
        except Exception as e:
            pytest.fail(f"generate_point_source_microlensing_magnitudes raised: {e}")
        mock_generate_maps.assert_called_once()
        assert isinstance(magnitudes, np.ndarray)
        assert magnitudes.shape == (num_images, len(time_array))
        assert np.issubdtype(magnitudes.dtype, np.floating)
        assert not np.any(np.isnan(magnitudes)) and not np.any(np.isinf(magnitudes))

    @pytest.mark.parametrize(
        "morphology_key, kwargs_source",
        [("gaussian", "kwargs_source_gaussian"), ("agn", "kwargs_source_agn_wave")],
    )
    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_microlensing_magnitudes_scalar_time(
        self,
        mock_generate_maps,
        base_init_kwargs,
        microlensing_params,
        kwargs_magnification_map_settings,
        morphology_key,
        kwargs_source,
        request,
    ):
        mock_generate_maps.return_value = create_mock_magmap_list(
            microlensing_params, kwargs_magnification_map_settings
        )
        kwargs_morphology = request.getfixturevalue(kwargs_source)
        num_images = len(microlensing_params["kappa_star"])
        scalar_time = 500.0

        args = base_init_kwargs.copy()
        args["point_source_morphology"] = morphology_key
        args["kwargs_source_morphology"] = kwargs_morphology

        ml_model = MicrolensingLightCurveFromLensModel(**args)
        add_safe_track(ml_model)

        try:
            magnitudes = ml_model.generate_point_source_microlensing_magnitudes(
                scalar_time
            )
        except Exception as e:
            pytest.fail(f"generate_..._magnitudes raised: {e}")
        mock_generate_maps.assert_called_once()
        assert isinstance(magnitudes, np.ndarray)
        assert magnitudes.shape == (num_images,)
        assert np.issubdtype(magnitudes.dtype, np.floating)
        assert not np.any(np.isnan(magnitudes)) and not np.any(np.isinf(magnitudes))

    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_microlensing_magnitudes_invalid_time(
        self, mock_generate_maps, ml_lens_model
    ):
        """Tests error handling for invalid time input."""
        mock_generate_maps.return_value = [MagicMock()]  # Dummy return
        with pytest.raises(ValueError, match="Time array not provided"):
            ml_lens_model.generate_point_source_microlensing_magnitudes(time="invalid")

    def test_properties_access(
        self,
        ml_lens_model,
        microlensing_params,
        kwargs_magnification_map_settings,
    ):
        """Test property access for lightcurves, tracks, and magmaps_images."""
        add_safe_track(ml_lens_model)

        # Test AttributeError when properties are accessed before generation
        with pytest.raises(AttributeError, match="Lightcurves are not set"):
            _ = ml_lens_model.lightcurves

        with pytest.raises(AttributeError, match="Tracks are not set"):
            _ = ml_lens_model.tracks

        with pytest.raises(AttributeError, match="Magnification maps are not set"):
            _ = ml_lens_model.magmaps_images

        # Generate data to populate the properties
        time_array = np.linspace(0, 1000, 50)

        with patch.object(
            MicrolensingLightCurveFromLensModel,
            "generate_magnification_maps_from_microlensing_params",
        ) as mock_generate_maps:
            mock_map_list = create_mock_magmap_list(
                microlensing_params, kwargs_magnification_map_settings
            )
            mock_generate_maps.return_value = mock_map_list

            # This should populate _lightcurves and _tracks
            _ = ml_lens_model.generate_point_source_microlensing_magnitudes(time_array)
            # manually set magmaps as the above call mocks its generation
            ml_lens_model._magmaps_images = mock_map_list

        # Now test that properties work correctly
        lightcurves = ml_lens_model.lightcurves
        assert isinstance(lightcurves, np.ndarray)
        assert len(lightcurves) == len(microlensing_params["kappa_star"])

        tracks = ml_lens_model.tracks
        assert isinstance(tracks, list)
        assert len(tracks) == len(microlensing_params["kappa_star"])

        magmaps = ml_lens_model.magmaps_images
        assert isinstance(magmaps, list)
        assert len(magmaps) == len(microlensing_params["kappa_star"])
