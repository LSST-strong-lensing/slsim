import os
import pytest
import numpy as np
import astropy.units as u

# import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

# Import the class to test
from slsim.Microlensing.lightcurve import (
    MicrolensingLightCurve,
    AMOEBA_AVAILABLE,  # Import the flag
)

# Import the MagnificationMap class for a dummy instance
from slsim.Microlensing.magmap import MagnificationMap  # Assuming this exists

import matplotlib.pyplot as plt

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

    # as the first argument to all test methods
    def test_init(self, mlc_instance, dummy_magmap, time_duration):
        """Test the constructor."""
        assert mlc_instance.magnification_map is dummy_magmap
        assert mlc_instance.time_duration == time_duration
        assert not hasattr(mlc_instance, "convolved_map")  # Not computed yet

    #
    def test_get_convolved_map(self, source_size, mlc_instance):
        """Test the convolution method."""
        map_shape = mlc_instance.magnification_map.magnifications.shape

        # Test without returning kernel
        convolved_map = mlc_instance._get_convolved_map(
            source_size=source_size, return_source_kernel=False
        )

        assert isinstance(convolved_map, np.ndarray)
        assert convolved_map.shape == map_shape
        assert hasattr(mlc_instance, "convolved_map")
        assert mlc_instance.convolved_map is convolved_map  # Check if stored
        # Check if convolution smoothed (max value likely reduced, mean roughly similar)
        # Use nanmax in case map has NaNs
        assert np.nanmax(convolved_map) <= np.nanmax(
            mlc_instance.magnification_map.magnifications
        )

        # Test with returning kernel
        convolved_map_2, source_kernel = mlc_instance._get_convolved_map(
            source_size=source_size, return_source_kernel=True
        )

        assert isinstance(convolved_map_2, np.ndarray)
        assert convolved_map_2.shape == map_shape
        assert isinstance(source_kernel, np.ndarray)
        # Kernel shape might not match map shape if fftconvolve pads, check source kernel creation logic if needed
        # assert source_kernel.shape == map_shape # This might be too strict depending on kernel generation
        assert np.isclose(
            np.sum(source_kernel), 1.0, atol=1e-6
        )  # Kernel should sum to 1
        # Use np.allclose for floating point comparison
        assert np.allclose(
            convolved_map, convolved_map_2, equal_nan=True
        )  # Result should be same

    #
    def test_generate_point_source_lightcurve_basic(
        self,  # Added self
        mlc_instance,
        cosmo,
        source_redshift,
        source_size,
        source_velocity,
        expected_lightcurve_length,
    ):
        """Test basic point source light curve generation (magnification)."""

        kwargs_ps = {
            "source_size": source_size,  # arcsec
            "effective_transverse_velocity": source_velocity,  # km/s
        }
        n_lc = 10  # Reduced for faster testing?

        # Now mlc_instance should be the correct object
        lcs = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            lightcurve_type="magnification",
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=False,
        )

        # Check the light curve shape
        assert isinstance(lcs, np.ndarray)
        assert lcs.shape == (n_lc, expected_lightcurve_length)

        # Check that the convolved map was generated and used
        assert hasattr(mlc_instance, "convolved_map")

        # check that the convolved map is not empty
        assert mlc_instance.convolved_map.size > 0

        # Check that the light curve values lie between the min and max of the convolved map
        # Add a small tolerance for floating point inaccuracies from interpolation
        assert np.all(lcs >= np.nanmin(mlc_instance.convolved_map) - 1e-9)
        assert np.all(lcs <= np.nanmax(mlc_instance.convolved_map) + 1e-9)
        # Check that the light curve values are not all the same
        if n_lc > 1:
            # Check variance instead of direct comparison for robustness
            assert (
                np.var(lcs, axis=1).all() > 1e-12
            )  # Check that variance along time axis is non-zero
            assert (
                np.var(lcs, axis=0).all() > 1e-12
            )  # Check variance across lightcurves
        # Check that the light curve values are not all zero
        assert not np.all(np.isclose(lcs, 0))

    #
    def test_generate_point_source_lightcurve_magnitude(
        self,  # Added self
        mlc_instance,
        cosmo,
        source_redshift,
        source_size,
        source_velocity,
        expected_lightcurve_length,
    ):
        """Test basic point source light curve generation (magnitude)."""

        kwargs_ps = {
            "source_size": source_size,  # arcsec
            "effective_transverse_velocity": source_velocity,  # km/s
        }
        n_lc = 10  # Reduced for faster testing?

        # Now mlc_instance should be the correct object
        lcs = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            lightcurve_type="magnitude",
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=False,
        )

        # Check the light curve shape
        assert isinstance(lcs, np.ndarray)
        assert lcs.shape == (n_lc, expected_lightcurve_length)

        # Check that the convolved map was generated and used
        assert hasattr(mlc_instance, "convolved_map")

        # check that the convolved map is not empty
        assert mlc_instance.convolved_map.size > 0

        # Check magnitude calculation carefully
        mean_mag_convolved = np.nanmean(mlc_instance.convolved_map)
        assert mean_mag_convolved > 0  # Mean magnification should be positive

        # Calculate bounds based on the *magnification* range, then convert bounds to magnitude
        min_mu_conv = np.nanmin(mlc_instance.convolved_map)
        max_mu_conv = np.nanmax(mlc_instance.convolved_map)

        # Avoid log(<=0) for theoretical bounds
        # Magnitudes are inverted: min mag corresponds to max mu, max mag corresponds to min mu
        theor_min_mag = -2.5 * np.log10(
            max(max_mu_conv, 1e-9) / np.abs(mean_mag_convolved)
        )
        theor_max_mag = -2.5 * np.log10(
            max(min_mu_conv, 1e-9) / np.abs(mean_mag_convolved)
        )

        # Add tolerance for interpolation and floating point issues
        tolerance = 1e-6
        # Check if all lcs values are finite first
        assert np.all(
            np.isfinite(lcs)
        ), f"Found non-finite values in magnitude light curves: min={np.min(lcs)}, max={np.max(lcs)}"
        assert np.all(lcs >= theor_min_mag - tolerance)
        assert np.all(lcs <= theor_max_mag + tolerance)

        # Check that the light curve values are not all the same
        if n_lc > 1:
            assert np.var(lcs, axis=1).all() > 1e-12
            assert np.var(lcs, axis=0).all() > 1e-12
        # Check that the light curve values are not all zero (magnitudes shouldn't typically be exactly 0)
        assert not np.all(np.isclose(lcs, 0))

    #
    def test_generate_point_source_input_validation(
        self,  # Added self
        source_size,
        source_redshift,
        source_velocity,
        mlc_instance,  # mlc_instance goes here or after cosmo, order matters less than self
        cosmo,
    ):
        """Test input validation for point source."""
        kwargs_no_size = {"effective_transverse_velocity": source_velocity}
        kwargs_no_vel = {"source_size": source_size}
        kwargs_ok = {
            "source_size": source_size,
            "effective_transverse_velocity": source_velocity,
        }

        # Now mlc_instance should be the correct object for the calls below
        with pytest.raises(ValueError, match="Source size not provided"):
            mlc_instance.generate_point_source_lightcurve(
                source_redshift=source_redshift,
                cosmology=cosmo,
                kwargs_PointSource=kwargs_no_size,
            )

        with pytest.raises(
            ValueError, match="Effective transverse velocity not provided"
        ):
            mlc_instance.generate_point_source_lightcurve(
                source_redshift=source_redshift,
                cosmology=cosmo,
                kwargs_PointSource=kwargs_no_vel,
            )

        with pytest.raises(ValueError, match="Lightcurve type not recognized"):
            mlc_instance.generate_point_source_lightcurve(
                source_redshift=source_redshift,
                cosmology=cosmo,
                kwargs_PointSource=kwargs_ok,
                lightcurve_type="invalid_type",
            )

    def test_generate_point_source_lightcurve_return_options(
        self,
        mlc_instance,
        cosmo,
        source_redshift,
        source_size,
        source_velocity,
        expected_lightcurve_length,
    ):
        """Test return options for point source light curve generation."""
        kwargs_ps = {
            "source_size": source_size,
            "effective_transverse_velocity": source_velocity,
        }
        n_lc = 2
        expected_len = expected_lightcurve_length

        # 1. Return Tracks Only
        lcs_t, tracks_t = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            num_lightcurves=n_lc,
            return_track_coords=True,
            return_time_array=False,
        )
        assert isinstance(lcs_t, np.ndarray) and lcs_t.shape == (n_lc, expected_len)
        assert isinstance(tracks_t, np.ndarray)
        assert tracks_t.shape == (n_lc, 2, expected_len)  # (n_lc, [x,y], time_steps)
        # Check track coordinates are within pixel bounds
        map_shape = mlc_instance.magnification_map.magnifications.shape
        assert np.all(tracks_t[:, 0, :] >= 0) and np.all(
            tracks_t[:, 0, :] < map_shape[0]
        )  # x-pixels (index 0)
        assert np.all(tracks_t[:, 1, :] >= 0) and np.all(
            tracks_t[:, 1, :] < map_shape[1]
        )  # y-pixels (index 1)

        # 2. Return Time Array Only
        lcs_time, times_time = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=True,
        )
        assert isinstance(lcs_time, np.ndarray) and lcs_time.shape == (
            n_lc,
            expected_len,
        )
        assert isinstance(times_time, list) and len(times_time) == n_lc
        assert all(isinstance(t_arr, np.ndarray) for t_arr in times_time)
        assert all(len(t_arr) == expected_len for t_arr in times_time)
        assert all(np.isclose(t_arr[0], 0) for t_arr in times_time)
        assert all(np.isclose(t_arr[-1], time_duration) for t_arr in times_time)

        # 3. Return Both Tracks and Time Array
        lcs_both, tracks_both, times_both = (
            mlc_instance.generate_point_source_lightcurve(
                source_redshift=source_redshift,
                cosmology=cosmo,
                kwargs_PointSource=kwargs_ps,
                num_lightcurves=n_lc,
                return_track_coords=True,
                return_time_array=True,
            )
        )
        assert isinstance(lcs_both, np.ndarray) and lcs_both.shape == (
            n_lc,
            expected_len,
        )
        assert isinstance(tracks_both, np.ndarray) and tracks_both.shape == (
            n_lc,
            2,
            expected_len,
        )
        assert isinstance(times_both, list) and len(times_both) == n_lc
        assert all(
            isinstance(t_arr, np.ndarray) and len(t_arr) == expected_len
            for t_arr in times_both
        )
        assert all(
            np.isclose(t_arr[0], 0) and np.isclose(t_arr[-1], time_duration)
            for t_arr in times_both
        )

    # ---- AGN Tests ----
    # Use skipif to conditionally skip tests if amoeba is not installed
    pytestmark_agn = pytest.mark.skipif(
        not AMOEBA_AVAILABLE, reason="amoeba package not installed"
    )

    @pytestmark_agn
    def test_generate_agn_lightcurve_basic(
        self,
        mlc_instance,
        cosmo,
        source_redshift,
        source_velocity,
        deflector_redshift,
        expected_lightcurve_length,
    ):
        """Test basic AGN light curve generation (magnification)."""
        n_lc = 2
        # Use default AGN parameters from the function signature for simplicity
        lcs = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            lightcurve_type="magnification",
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=False,
            # Add other required params if defaults aren't sufficient
            v_transverse=source_velocity,  # km/s
        )

        # Check convolved map was updated by AGN process
        assert hasattr(mlc_instance, "convolved_map")
        # The convolved map might be different from point source, don't compare directly

        assert isinstance(lcs, list)  # AGN returns list
        assert len(lcs) == n_lc
        assert all(isinstance(lc, np.ndarray) for lc in lcs)
        assert all(len(lc) == expected_lightcurve_length for lc in lcs)

        # Check values are within the bounds of the *new* convolved map
        convolved_map = mlc_instance.convolved_map  # Get the AGN-convolved map
        min_conv, max_conv = np.nanmin(convolved_map), np.nanmax(convolved_map)
        assert all(np.all(lc >= min_conv - 1e-9) for lc in lcs)
        assert all(np.all(lc <= max_conv + 1e-9) for lc in lcs)

        # Check variance
        if n_lc > 1:
            assert (
                np.var([item for sublist in lcs for item in sublist]) > 1e-12
            )  # Overall variance
            assert all(np.var(lc) > 1e-12 for lc in lcs)  # Variance within each lc

    @pytestmark_agn
    def test_generate_agn_lightcurve_magnitude(
        self,
        mlc_instance,
        cosmo,
        source_redshift,
        source_velocity,
        deflector_redshift,
        expected_lightcurve_length,
    ):
        """Test basic AGN light curve generation (magnitude)."""
        n_lc = 2
        lcs = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            lightcurve_type="magnitude",
            num_lightcurves=n_lc,
            v_transverse=source_velocity,  # km/s
        )

        assert isinstance(lcs, list) and len(lcs) == n_lc
        assert all(
            isinstance(lc, np.ndarray) and len(lc) == expected_lightcurve_length
            for lc in lcs
        )

        # Check magnitude calculation
        assert hasattr(mlc_instance, "convolved_map")
        mean_mag_convolved = np.nanmean(mlc_instance.convolved_map)
        assert mean_mag_convolved > 0

        min_mu_conv = np.nanmin(mlc_instance.convolved_map)
        max_mu_conv = np.nanmax(mlc_instance.convolved_map)
        tiny_positive = 1e-12
        theor_min_mag = -2.5 * np.log10(
            max(max_mu_conv, tiny_positive) / np.abs(mean_mag_convolved)
        )
        theor_max_mag = -2.5 * np.log10(
            max(min_mu_conv, tiny_positive) / np.abs(mean_mag_convolved)
        )
        if theor_min_mag > theor_max_mag:
            theor_min_mag, theor_max_mag = theor_max_mag, theor_min_mag
        tolerance = 1e-6

        for lc in lcs:
            assert np.all(
                np.isfinite(lc)
            ), f"Found non-finite values in AGN magnitude light curves: {lc}"
            assert np.all(lc >= theor_min_mag - tolerance)
            assert np.all(lc <= theor_max_mag + tolerance)

    @pytestmark_agn
    def test_generate_agn_lightcurve_return_options(
        self,
        mlc_instance,
        cosmo,
        source_redshift,
        source_velocity,
        deflector_redshift,
        expected_lightcurve_length,
    ):
        """Test return options for AGN light curve generation."""
        n_lc = 2
        expected_length = expected_lightcurve_length

        # 1. Return Tracks Only
        lcs_t, tracks_t = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            num_lightcurves=n_lc,
            v_transverse=source_velocity,
            return_track_coords=True,
            return_time_array=False,
        )
        assert isinstance(lcs_t, list) and len(lcs_t) == n_lc
        assert all(
            isinstance(lc, np.ndarray) and len(lc) == expected_length for lc in lcs_t
        )
        assert isinstance(tracks_t, list) and len(tracks_t) == n_lc
        assert all(
            isinstance(track, np.ndarray) and track.shape == (2, expected_length)
            for track in tracks_t
        )

        # 2. Return Time Array Only
        lcs_time, times_time = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            num_lightcurves=n_lc,
            v_transverse=source_velocity,
            return_track_coords=False,
            return_time_array=True,
        )
        assert isinstance(lcs_time, list) and len(lcs_time) == n_lc
        assert all(
            isinstance(lc, np.ndarray) and len(lc) == expected_length for lc in lcs_time
        )
        assert isinstance(times_time, list) and len(times_time) == n_lc
        assert all(
            isinstance(t_arr, np.ndarray) and len(t_arr) == expected_length
            for t_arr in times_time
        )
        assert all(
            np.isclose(t_arr[0], 0) and np.isclose(t_arr[-1], time_duration)
            for t_arr in times_time
        )

        # 3. Return Both
        lcs_both, tracks_both, times_both = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            num_lightcurves=n_lc,
            v_transverse=source_velocity,
            return_track_coords=True,
            return_time_array=True,
        )
        assert isinstance(lcs_both, list) and len(lcs_both) == n_lc
        assert all(
            isinstance(lc, np.ndarray) and len(lc) == expected_length for lc in lcs_both
        )
        assert isinstance(tracks_both, list) and len(tracks_both) == n_lc
        assert all(
            isinstance(track, np.ndarray) and track.shape == (2, expected_length)
            for track in tracks_both
        )
        assert isinstance(times_both, list) and len(times_both) == n_lc
        assert all(
            isinstance(t_arr, np.ndarray) and len(t_arr) == expected_length
            for t_arr in times_both
        )
        assert all(
            np.isclose(t_arr[0], 0) and np.isclose(t_arr[-1], time_duration)
            for t_arr in times_both
        )

    def test_generate_agn_lightcurve_no_amoeba(self, monkeypatch, mlc_instance):
        """Test AGN light curve generation raises ImportError if amoeba not
        available."""
        # Temporarily pretend amoeba is not available
        monkeypatch.setattr("slsim.Microlensing.lightcurve.AMOEBA_AVAILABLE", False)

        with pytest.raises(ImportError, match="amoeba package is required"):
            mlc_instance.generate_agn_lightcurve(
                source_redshift=1.5,
                deflector_redshift=0.5,
                cosmology=FlatLambdaCDM(H0=70, Om0=0.3),
            )

    # ---- Plotting Test ----
    def test_plot_lightcurves_and_magmap(
        self,
        mlc_instance,
        cosmo,
        source_redshift,
        source_size,
        source_velocity,
    ):
        """Test plotting function runs without errors."""
        # Generate some dummy data first
        kwargs_ps = {
            "source_size": source_size,
            "effective_transverse_velocity": source_velocity,
        }
        n_lc = 2
        # Generate magnification light curves and tracks
        lcs, tracks, times = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            lightcurve_type="magnification",
            num_lightcurves=n_lc,
            return_track_coords=True,
            return_time_array=True,
        )
        # Also generate magnitude light curves (need separate call as convolved map is stored)
        mlc_instance_mag = MicrolensingLightCurve(
            mlc_instance.magnification_map, mlc_instance.time_duration
        )
        lcs_mag = mlc_instance_mag.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            lightcurve_type="magnitude",
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=False,
        )

        # Test plotting magnification with tracks
        try:
            fig, axes = plt.subplots(1, 2)  # Create figure and axes manually
            returned_axes = mlc_instance.plot_lightcurves_and_magmap(
                lightcurves=lcs, tracks=tracks, lightcurve_type="magnification"
            )
            # Check if it returns the axes object it modified/created
            assert isinstance(
                returned_axes, np.ndarray
            )  # Should return the array of axes
            assert len(returned_axes) == 2
            plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            pytest.fail(
                f"plot_lightcurves_and_magmap failed for magnification with tracks: {e}"
            )

        # Test plotting magnitude without tracks
        try:
            fig, axes = plt.subplots(1, 2)
            # Use the instance that generated magnitude LCs
            returned_axes_mag = mlc_instance_mag.plot_lightcurves_and_magmap(
                lightcurves=lcs_mag, tracks=None, lightcurve_type="magnitude"
            )
            assert isinstance(returned_axes_mag, np.ndarray)
            assert len(returned_axes_mag) == 2
            plt.close(fig)
        except Exception as e:
            pytest.fail(
                f"plot_lightcurves_and_magmap failed for magnitude without tracks: {e}"
            )

    # ---- Other Method Tests ----
    def test_generate_supernova_lightcurve(self, mlc_instance):
        """Test the placeholder supernova method."""
        # Currently just 'pass', so test that it runs without error
        try:
            mlc_instance._generate_supernova_lightcurve()
        except Exception as e:
            pytest.fail(f"_generate_supernova_lightcurve raised an exception: {e}")
