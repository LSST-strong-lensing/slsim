import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM

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
    num_pix = 1000  # Resolution of saved maps
    half_len = 25  # Extent (in theta_star units) of saved maps
    return {
        "theta_star": theta_star,
        "center_x": 0,
        "center_y": 0,
        "half_length_x": half_len * theta_star,
        "half_length_y": half_len * theta_star,
        "mass_function": "kroupa",
        "m_solar": 1.0,
        "m_lower": 0.08,
        "m_upper": 100,
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
def ml_lens_model():
    """Provides an instance of the class under test."""
    return MicrolensingLightCurveFromLensModel()


# ---- Helper Function to Load Maps and Create Mock Return Value ---
def create_mock_magmap_list(microlensing_params, kwargs_magnification_map_settings):
    """Loads saved magnification maps and creates MagnificationMap
    instances."""
    num_images = len(microlensing_params["kappa_star"])
    loaded_magmaps = []

    # --- Robust Path Finding ---
    test_dir = os.path.dirname(os.path.abspath(__file__))
    # Try relative to test file first (e.g., tests/TestData/...)
    saved_map_directory_rel = os.path.join(
        test_dir, "..", "TestData", "test_data_microlensing", "saved_magmaps2D"
    )
    # Try relative to parent of test dir (e.g., slsim/TestData/...)
    base_dir = os.path.dirname(test_dir)
    saved_map_directory_root = os.path.join(
        base_dir, "TestData", "test_data_microlensing", "saved_magmaps2D"
    )

    if os.path.isdir(saved_map_directory_rel):
        saved_map_directory = saved_map_directory_rel
    elif os.path.isdir(saved_map_directory_root):
        saved_map_directory = saved_map_directory_root
    else:
        pytest.fail(
            f"Could not find saved map directory at {saved_map_directory_rel} or {saved_map_directory_root}"
        )
    # --- End Robust Path Finding ---

    for i in range(num_images):
        map_filename = f"magmap_image_{i}.npy"
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
            # REMOVED mu_ave assignment as per user request
            # valid_mu = mag_data[np.isfinite(mag_data) & (mag_data != 0)]
            # magmap_obj.mu_ave = np.mean(valid_mu) if len(valid_mu) > 0 else 1.0
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

    @pytest.mark.parametrize("magmap_frame", [True, False])
    def test_effective_transverse_velocity_images(
        self,
        ml_lens_model,
        lens_source_info,
        microlensing_params,
        cosmology,
        magmap_frame,
    ):
        num_images = len(microlensing_params["shear_phi"])
        velocities, angles = ml_lens_model.effective_transverse_velocity_images(
            lens_source_info["source_redshift"],
            lens_source_info["deflector_redshift"],
            lens_source_info["ra_lens"],
            lens_source_info["dec_lens"],
            cosmology,
            microlensing_params["shear_phi"],
            lens_source_info["deflector_velocity_dispersion"],
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
        from astropy import units as u

        ra_q = (
            lens_source_info["ra_lens"] * u.deg
        )  # now a Quantity → hits `else: ra_l = ra_lens`
        dec_q = (
            lens_source_info["dec_lens"] * u.deg
        )  # now a Quantity → hits `else: dec_l = dec_lens`
        sigma_q = lens_source_info["deflector_velocity_dispersion"] * u.km / u.s
        velocities_q, angles_q = ml_lens_model.effective_transverse_velocity_images(
            lens_source_info["source_redshift"],
            lens_source_info["deflector_redshift"],
            ra_q,
            dec_q,
            cosmology,
            microlensing_params["shear_phi"],
            sigma_q,  # Quantity → hits `else: sig_star = …`
            random_seed=42,
            magmap_reference_frame=magmap_frame,
        )
        assert velocities_q.shape == (num_images,)
        # ─────────────────────────────────────────────────────────────────────────────

        # ────── COVER the e1 ZERO‐VECTOR BRANCH ──────
        # Choose dec_lens = 90° so u_los is (0,0,1) and first cross yields zero
        ra_pole = 0 * u.deg
        dec_pole = 90 * u.deg
        v_pole, a_pole = ml_lens_model.effective_transverse_velocity_images(
            lens_source_info["source_redshift"],
            lens_source_info["deflector_redshift"],
            ra_pole,
            dec_pole,
            cosmology,
            microlensing_params["shear_phi"],
            lens_source_info["deflector_velocity_dispersion"],
            random_seed=42,
            magmap_reference_frame=magmap_frame,
        )
        assert isinstance(v_pole, np.ndarray)
        # ─────────────────────────────────────────────────────────────────────────

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
        num_images = len(microlensing_params["kappa_star"])
        mock_map_list = create_mock_magmap_list(
            microlensing_params, kwargs_magnification_map_settings
        )
        assert isinstance(mock_map_list, list)
        assert len(mock_map_list) == num_images
        for i, magmap_obj in enumerate(mock_map_list):
            assert isinstance(magmap_obj, MagnificationMap)
            assert hasattr(magmap_obj, "magnifications")
            assert magmap_obj.magnifications is not None
            assert magmap_obj.kappa_tot == microlensing_params["kappa_tot"][i]
            assert magmap_obj.shear == microlensing_params["shear"][i]
            assert magmap_obj.kappa_star == microlensing_params["kappa_star"][i]
            assert (
                magmap_obj.theta_star == kwargs_magnification_map_settings["theta_star"]
            )
            assert (
                magmap_obj.num_pixels_x
                == kwargs_magnification_map_settings["num_pixels_x"]
            )
            # Removed mu_ave check as requested
            # assert hasattr(magmap_obj, 'mu_ave') and isinstance(magmap_obj.mu_ave, (float, np.floating))

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
        ml_lens_model,
        microlensing_params,
        lens_source_info,
        cosmology,
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
        time_array = np.linspace(0, 1000, 100)
        try:
            lightcurves, tracks, time_arrays = (
                ml_lens_model.generate_point_source_lightcurves(
                    time_array,
                    lens_source_info["source_redshift"],
                    lens_source_info["deflector_redshift"],
                    microlensing_params["kappa_star"],
                    microlensing_params["kappa_tot"],
                    microlensing_params["shear"],
                    microlensing_params["shear_phi"],
                    lens_source_info["ra_lens"],
                    lens_source_info["dec_lens"],
                    lens_source_info["deflector_velocity_dispersion"],
                    cosmology,
                    kwargs_magnification_map_settings,
                    morphology_key,
                    kwargs_morphology,
                    lightcurve_type,
                    num_lc,
                )
            )
        except Exception as e:
            pytest.fail(f"generate_point_source_lightcurves raised: {e}")
        mock_generate_maps.assert_called_once_with(
            kappa_star_images=microlensing_params["kappa_star"],
            kappa_tot_images=microlensing_params["kappa_tot"],
            shear_images=microlensing_params["shear"],
            kwargs_MagnificationMap=kwargs_magnification_map_settings,
        )
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
        )  # Check track has some length, but not necessarily same as interpolated LC
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
        self, mock_generate_maps, ml_lens_model, cosmology
    ):
        """Tests error handling for invalid time input."""
        # Set a dummy return for the mocked map generation, although it shouldn't be used much
        mock_generate_maps.return_value = [MagicMock()]
        with pytest.raises(ValueError, match="Time array not provided"):
            ml_lens_model.generate_point_source_lightcurves(
                time="invalid",
                source_redshift=1,
                deflector_redshift=0.5,
                kappa_star_images=[0.1],
                kappa_tot_images=[0.4],
                shear_images=[0.1],
                shear_phi_angle_images=[0],
                ra_lens=0,
                dec_lens=0,
                deflector_velocity_dispersion=200,
                cosmology=cosmology,  # Pass fixture instance
                kwargs_MagnificationMap={},
                point_source_morphology="gaussian",
                kwargs_source_morphology={},
            )

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
        ml_lens_model,
        microlensing_params,
        lens_source_info,
        cosmology,
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
        time_array = np.linspace(0, 1000, 50)
        try:
            magnitudes = ml_lens_model.generate_point_source_microlensing_magnitudes(
                time_array,
                lens_source_info["source_redshift"],
                lens_source_info["deflector_redshift"],
                microlensing_params["kappa_star"],
                microlensing_params["kappa_tot"],
                microlensing_params["shear"],
                microlensing_params["shear_phi"],
                lens_source_info["ra_lens"],
                lens_source_info["dec_lens"],
                lens_source_info["deflector_velocity_dispersion"],
                cosmology,
                kwargs_magnification_map_settings,
                morphology_key,
                kwargs_morphology,
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
        ml_lens_model,
        microlensing_params,
        lens_source_info,
        cosmology,
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
        time_array = np.linspace(0, 1000, 50)
        time_array = time_array.tolist()  # Convert to list
        try:
            magnitudes = ml_lens_model.generate_point_source_microlensing_magnitudes(
                time_array,
                lens_source_info["source_redshift"],
                lens_source_info["deflector_redshift"],
                microlensing_params["kappa_star"],
                microlensing_params["kappa_tot"],
                microlensing_params["shear"],
                microlensing_params["shear_phi"],
                lens_source_info["ra_lens"],
                lens_source_info["dec_lens"],
                lens_source_info["deflector_velocity_dispersion"],
                cosmology,
                kwargs_magnification_map_settings,
                morphology_key,
                kwargs_morphology,
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
    def test_generate_point_source_microlensing_magnitudes_scalar_time(
        self,
        mock_generate_maps,
        ml_lens_model,
        microlensing_params,
        lens_source_info,
        cosmology,
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
        try:
            magnitudes = ml_lens_model.generate_point_source_microlensing_magnitudes(
                scalar_time,
                lens_source_info["source_redshift"],
                lens_source_info["deflector_redshift"],
                microlensing_params["kappa_star"],
                microlensing_params["kappa_tot"],
                microlensing_params["shear"],
                microlensing_params["shear_phi"],
                lens_source_info["ra_lens"],
                lens_source_info["dec_lens"],
                lens_source_info["deflector_velocity_dispersion"],
                cosmology,
                kwargs_magnification_map_settings,
                morphology_key,
                kwargs_morphology,
            )
        except Exception as e:
            pytest.fail(f"generate_..._magnitudes raised: {e}")
        mock_generate_maps.assert_called_once()
        assert isinstance(magnitudes, np.ndarray)
        assert magnitudes.shape == (num_images,)
        assert np.issubdtype(magnitudes.dtype, np.floating)
        assert not np.any(np.isnan(magnitudes)) and not np.any(np.isinf(magnitudes))

    # Patch map generation even for invalid time test to avoid IPM errors
    @patch.object(
        MicrolensingLightCurveFromLensModel,
        "generate_magnification_maps_from_microlensing_params",
    )
    def test_generate_point_source_microlensing_magnitudes_invalid_time(
        self, mock_generate_maps, ml_lens_model, cosmology
    ):
        """Tests error handling for invalid time input."""
        mock_generate_maps.return_value = [MagicMock()]  # Dummy return
        with pytest.raises(ValueError, match="Time array not provided"):
            ml_lens_model.generate_point_source_microlensing_magnitudes(
                time="invalid",
                source_redshift=1,
                deflector_redshift=0.5,
                kappa_star_images=[0.1],
                kappa_tot_images=[0.4],
                shear_images=[0.1],
                shear_phi_angle_images=[0],
                ra_lens=0,
                dec_lens=0,
                deflector_velocity_dispersion=200,
                cosmology=cosmology,
                kwargs_MagnificationMap={},
                point_source_morphology="gaussian",
                kwargs_source_morphology={},
            )
