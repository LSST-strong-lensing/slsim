import os
import pytest
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

# Import the class to test
from slsim.Microlensing.lightcurve import (
    MicrolensingLightCurve,
)

# Import supporting classes and functions
from slsim.Microlensing.magmap import MagnificationMap
from slsim.Microlensing.source_morphology import (
    GaussianSourceMorphology,
    AGNSourceMorphology,
)

# Import dependencies directly as they are assumed available
import speclite.filters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from slsim.Util.astro_util import (
    calculate_accretion_disk_emission,
    calculate_gravitational_radius,
    extract_light_curve,
)


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

    # Robust path handling (adjust if needed for your structure)
    try:
        # Assuming TestData is in the same directory as the test file
        test_dir = os.path.dirname(__file__)
        magmap2D_path = os.path.join(test_dir, "..", "TestData", "test_magmap2D.npy")
        if not os.path.exists(magmap2D_path):
            # Add more debug info for CI if it fails again
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
        assert ml_lc.magnification_map is magmap_instance
        assert ml_lc.time_duration == time_dur
        assert ml_lc.point_source_morphology == morphology
        assert ml_lc.kwargs_source_morphology == kwargs_source_morphology_Gaussian
        assert ml_lc.convolved_map is None
        assert ml_lc.source_morphology is None

    def test_get_convolved_map_gaussian(self, ml_lc_gaussian, magmap_instance):
        conv_map = ml_lc_gaussian.get_convolved_map(return_source_morphology=False)
        assert isinstance(conv_map, np.ndarray)
        assert conv_map.shape == magmap_instance.magnifications.shape
        assert ml_lc_gaussian.convolved_map is conv_map
        assert isinstance(ml_lc_gaussian.source_morphology, GaussianSourceMorphology)
        ml_lc_gaussian.convolved_map = None
        ml_lc_gaussian.source_morphology = None
        conv_map2, morph = ml_lc_gaussian.get_convolved_map(
            return_source_morphology=True
        )
        assert np.array_equal(conv_map, conv_map2)
        assert isinstance(morph, GaussianSourceMorphology)
        assert ml_lc_gaussian.source_morphology is morph

    @pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in divide:RuntimeWarning"
    )
    def test_get_convolved_map_agn_wave(self, ml_lc_agn_wave, magmap_instance):
        conv_map, morph = ml_lc_agn_wave.get_convolved_map(
            return_source_morphology=True
        )
        assert isinstance(conv_map, np.ndarray)
        assert conv_map.shape == magmap_instance.magnifications.shape
        assert ml_lc_agn_wave.convolved_map is conv_map
        assert isinstance(morph, AGNSourceMorphology)
        assert ml_lc_agn_wave.source_morphology is morph
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
        assert ml_lc_agn_band.convolved_map is conv_map
        assert isinstance(morph, AGNSourceMorphology)
        assert ml_lc_agn_band.source_morphology is morph
        assert hasattr(morph, "pixel_scale_m")

    def test_get_convolved_map_supernovae(self, ml_lc_gaussian):
        ml_lc_gaussian.point_source_morphology = "supernovae"
        with pytest.raises(NotImplementedError):
            ml_lc_gaussian.get_convolved_map()

    def test_get_convolved_map_invalid_type(self, ml_lc_gaussian):
        ml_lc_gaussian.point_source_morphology = "invalid_type"
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
            lcs = ml_lc_gaussian.generate_lightcurves(
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
            lcs = ml_lc_agn_wave.generate_lightcurves(
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

    @pytest.mark.parametrize(
        "ret_track, ret_time, expected_len",
        [(False, False, 1), (True, False, 2), (False, True, 2), (True, True, 3)],
    )
    def test_generate_lightcurves_return_options(
        self, ml_lc_gaussian, cosmology, ret_track, ret_time, expected_len
    ):
        """Tests different return combinations using real extract."""
        num_lc = 2
        try:
            result = ml_lc_gaussian.generate_lightcurves(
                source_redshift=0.5,
                cosmo=cosmology,
                num_lightcurves=num_lc,
                return_track_coords=ret_track,
                return_time_array=ret_time,
            )
        except Exception as e:
            pytest.fail(f"generate_lightcurves raised: {e}")
        if expected_len == 1:
            lcs = result
            assert isinstance(lcs, list) and len(lcs) == num_lc
        else:
            assert isinstance(result, tuple)
            assert len(result) == expected_len
            lcs = result[0]
            assert isinstance(lcs, list) and len(lcs) == num_lc
        if ret_track:
            tracks = result[1]
            assert isinstance(tracks, list) and len(tracks) == num_lc
            assert isinstance(tracks[0], np.ndarray)
            assert tracks[0].shape[0] == 2
            assert tracks[0].shape[1] == len(lcs[0])
        if ret_time:
            time_arrays = result[-1]
            assert isinstance(time_arrays, list) and len(time_arrays) == num_lc
            assert isinstance(time_arrays[0], np.ndarray)
            assert len(time_arrays[0]) == len(lcs[0])
            assert np.isclose(time_arrays[0][0], 0)
            assert np.isclose(time_arrays[0][-1], ml_lc_gaussian.time_duration)

    def test_generate_lightcurves_specific_start_and_angle(
        self, ml_lc_gaussian, cosmology
    ):
        """Tests passing specific start position and angle using real
        extract."""
        map_shape = ml_lc_gaussian.magnification_map.magnifications.shape
        x_start, y_start, phi = (
            map_shape[1] // 2 + 10,
            map_shape[0] // 2 - 5,
            45.0,
        )  # Offset slightly
        try:
            lcs = ml_lc_gaussian.generate_lightcurves(
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

    # --- Test Plotting (Execution Check Only) ---

    def test_plot_lightcurves_and_magmap_runs_magnitude(
        self, ml_lc_gaussian, cosmology
    ):
        """Tests plotting function runs without error (magnitude)."""
        num_lc = 2
        lcs, tracks = ml_lc_gaussian.generate_lightcurves(
            0.5, cosmology, num_lightcurves=num_lc, return_track_coords=True
        )
        ml_lc_gaussian.get_convolved_map()
        try:
            ax_return = ml_lc_gaussian.plot_lightcurves_and_magmap(
                lcs, tracks, "magnitude"
            )
            assert isinstance(ax_return, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in ax_return.flat)
        except Exception as e:
            pytest.fail(f"plot_lightcurves_and_magmap raised: {e}")
        finally:
            plt.close("all")

    @pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in divide:RuntimeWarning"
    )
    def test_plot_lightcurves_and_magmap_runs_magnification(
        self, ml_lc_agn_wave, cosmology
    ):
        """Tests plotting function runs without error (magnification)."""
        num_lc = 1
        lcs = ml_lc_agn_wave.generate_lightcurves(
            0.5, cosmology, num_lightcurves=num_lc, lightcurve_type="magnification"
        )
        ml_lc_agn_wave.get_convolved_map()
        try:
            ax_return = ml_lc_agn_wave.plot_lightcurves_and_magmap(
                lcs, tracks=None, lightcurve_type="magnification"
            )
            assert isinstance(ax_return, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in ax_return.flat)
        except Exception as e:
            pytest.fail(f"plot_lightcurves_and_magmap raised: {e}")
        finally:
            plt.close("all")
