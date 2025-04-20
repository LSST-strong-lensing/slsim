import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY  # ANY helps match arguments loosely
import astropy.units as u

# import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

# Import the class to test
from slsim.Microlensing.lightcurve import MicrolensingLightCurve, AMOEBA_AVAILABLE

# Import the MagnificationMap class for a dummy instance
from slsim.Microlensing.magmap import MagnificationMap  # Assuming this exists

# ---- Test Fixtures ----


@pytest.fixture
def theta_star():
    """Provides a theta_star value for testing."""
    return 4e-6  # arcsec


# Create a dummy MagnificationMap class for isolated testing
@pytest.fixture
def dummy_magmap():
    """Provides a basic MagnificationMap instance for testing."""

    magmap2D_path = os.path.join(
        os.path.dirname(__file__), "TestData", "test_magmap2D.npy"
    )

    # a precomputed map for the parameters below is available in the TestData folder
    magmap2D = np.load(magmap2D_path)
    kwargs_MagnificationMap = {
        "kappa_tot": 0.34960889,
        "shear": 0.34860889,
        "kappa_star": 0.24,
        "theta_star": theta_star(),
        "rectangular": True,
        "center_x": 0,  # arcsec
        "center_y": 0,  # arcsec
        "half_length_x": 25 * theta_star(),  # arcsec
        "half_length_y": 25 * theta_star(),  # arcsec
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
def mlc_instance():
    """Provides a MicrolensingLightCurve instance."""
    return MicrolensingLightCurve(
        magnification_map=dummy_magmap(), time_duration=time_duration()
    )


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

    def test_init(self, mlc_instance, dummy_magmap, time_duration):
        """Test the constructor."""
        assert mlc_instance.magnification_map is dummy_magmap
        assert mlc_instance.time_duration == time_duration
        assert not hasattr(mlc_instance, "convolved_map")  # Not computed yet

    def test_get_convolved_map(self, mlc_instance):
        """Test the convolution method."""
        source_size = 0.05  # arcsec
        map_shape = mlc_instance.magnification_map.magnifications.shape

        # Test without returning kernel
        convolved_map = mlc_instance._get_convolved_map(
            source_size=source_size, return_source_kernel=False
        )

        assert isinstance(convolved_map, np.ndarray)
        assert convolved_map.shape == map_shape
        assert convolved_map.dtype == np.float32  # Check dtype optimization
        assert hasattr(mlc_instance, "convolved_map")
        assert mlc_instance.convolved_map is convolved_map  # Check if stored
        # Check if convolution smoothed (max value likely reduced, mean roughly similar)
        assert np.max(convolved_map) <= np.max(
            mlc_instance.magnification_map.magnifications
        )

        # Test with returning kernel
        convolved_map_2, source_kernel = mlc_instance._get_convolved_map(
            source_size=source_size, return_source_kernel=True
        )

        assert isinstance(convolved_map_2, np.ndarray)
        assert convolved_map_2.shape == map_shape
        assert isinstance(source_kernel, np.ndarray)
        assert source_kernel.shape == map_shape
        assert np.isclose(
            np.sum(source_kernel), 1.0, atol=1e-6
        )  # Kernel should sum to 1
        assert np.all(convolved_map == convolved_map_2)  # Result should be same

    def test_generate_point_source_lightcurve_basic(
        mlc_instance,
        cosmo,
        source_redshift,
        source_size,
        source_velocity,
    ):
        """Test basic point source light curve generation (magnification)."""

        kwargs_ps = {
            "source_size": source_size,  # arcsec
            "effective_transverse_velocity": source_velocity,  # km/s
        }
        n_lc = 10

        lcs = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            lightcurve_type="magnification",
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=False,
        )

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
        expected_length = (
            int(source_velocity_mps * expected_time_seconds / pixel_size_meter) + 2
        )

        # Check the light curve shape
        assert isinstance(lcs, np.ndarray)
        assert lcs.shape == (n_lc, expected_length)

        # Check that the convolved map was generated and used
        assert hasattr(mlc_instance, "convolved_map")

        # check that the convolved map is not empty
        assert mlc_instance.convolved_map.size > 0

        # Check that the light curve values lie between the min and max of the convolved map
        assert np.all(lcs >= np.nanmin(mlc_instance.convolved_map))
        assert np.all(lcs <= np.nanmax(mlc_instance.convolved_map))
        # Check that the light curve values are not all the same
        if n_lc > 1:
            assert not np.all(lcs == lcs[0, :])
        # Check that the light curve values are not all zero
        assert not np.all(lcs == 0)

    def test_generate_point_source_lightcurve_magnitude(
        mlc_instance,
        cosmo,
        source_redshift,
        source_size,
        source_velocity,
    ):
        """Test basic point source light curve generation (magnitude)."""

        kwargs_ps = {
            "source_size": source_size,  # arcsec
            "effective_transverse_velocity": source_velocity,  # km/s
        }
        n_lc = 10

        lcs = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            lightcurve_type="magnitude",
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=False,
        )

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
        expected_length = (
            int(source_velocity_mps * expected_time_seconds / pixel_size_meter) + 2
        )

        # Check the light curve shape
        assert isinstance(lcs, np.ndarray)
        assert lcs.shape == (n_lc, expected_length)

        # Check that the convolved map was generated and used
        assert hasattr(mlc_instance, "convolved_map")

        # check that the convolved map is not empty
        assert mlc_instance.convolved_map.size > 0

        # Check that the light curve values lie between the min and max of the convolved map
        mean_mag_convolved = np.nanmean(mlc_instance.convolved_map)
        min_magnitude = -2.5 * np.log10(
            np.nanmax(mlc_instance.convolved_map) / np.abs(mean_mag_convolved)
        )
        max_magnitude = -2.5 * np.log10(
            np.nanmin(mlc_instance.convolved_map) / np.abs(mean_mag_convolved)
        )
        assert np.all(lcs >= min_magnitude)
        assert np.all(lcs <= max_magnitude)

        # Check that the light curve values are not all the same
        if n_lc > 1:
            assert not np.all(lcs == lcs[0, :])
        # Check that the light curve values are not all zero
        assert not np.all(lcs == 0)

    def test_generate_point_source_input_validation(
        self, source_size, source_redshift, source_velocity, mlc_instance, cosmo
    ):
        """Test input validation for point source."""
        kwargs_no_size = {"effective_transverse_velocity": source_velocity}
        kwargs_no_vel = {"source_size": source_size}
        kwargs_ok = {
            "source_size": source_size,
            "effective_transverse_velocity": source_velocity,
        }

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

    ###### DONE TILL HERE ######
