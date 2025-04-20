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

    def test_generate_point_source_input_validation(self, mlc_instance, cosmo):
        """Test input validation for point source."""
        source_redshift = 1.5
        kwargs_no_size = {"effective_transverse_velocity": 1000}
        kwargs_no_vel = {"source_size": 0.03}
        kwargs_ok = {"source_size": 0.03, "effective_transverse_velocity": 1000}

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

    @patch(
        "slsim.Microlensing.light_curve.extract_light_curve",
        side_effect=mock_extract_light_curve,
    )
    def test_generate_point_source_returns(self, mock_extract, mlc_instance, cosmo):
        """Test return options for point source light curves."""
        source_redshift = 1.5
        kwargs_ps = {"source_size": 0.03, "effective_transverse_velocity": 1000}
        n_lc = 2

        # Test return tracks
        lcs, tracks = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            num_lightcurves=n_lc,
            return_track_coords=True,
            return_time_array=False,
        )
        assert isinstance(lcs, np.ndarray)
        assert isinstance(tracks, np.ndarray)
        assert lcs.shape == (n_lc, MOCK_LC_LEN)
        assert tracks.shape == (n_lc, 2, MOCK_LC_LEN)  # num_lc, (x,y), time_steps
        assert mock_extract.call_count == n_lc

        mock_extract.reset_mock()

        # Test return time
        lcs, times = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=True,
        )
        assert isinstance(lcs, np.ndarray)
        assert isinstance(times, list)
        assert len(times) == n_lc
        assert isinstance(times[0], np.ndarray)
        assert len(times[0]) == MOCK_LC_LEN
        assert np.isclose(times[0][0], 0)
        assert np.isclose(times[0][-1], mlc_instance.time_duration)
        assert mock_extract.call_count == n_lc

        mock_extract.reset_mock()

        # Test return tracks and time
        lcs, tracks, times = mlc_instance.generate_point_source_lightcurve(
            source_redshift=source_redshift,
            cosmology=cosmo,
            kwargs_PointSource=kwargs_ps,
            num_lightcurves=n_lc,
            return_track_coords=True,
            return_time_array=True,
        )
        assert isinstance(lcs, np.ndarray)
        assert isinstance(tracks, np.ndarray)
        assert isinstance(times, list)
        assert lcs.shape == (n_lc, MOCK_LC_LEN)
        assert tracks.shape == (n_lc, 2, MOCK_LC_LEN)
        assert len(times) == n_lc
        assert len(times[0]) == MOCK_LC_LEN
        assert mock_extract.call_count == n_lc

    # --- AGN Tests ---

    # Mock amoeba classes and functions needed
    # We need to mock them *before* the class definition potentially imports them
    # Using patch decorators on the test methods is usually the easiest way.

    # Create mock objects for amoeba classes
    mock_amoeba_disk = MagicMock()
    mock_disk_projection = MagicMock()
    mock_disk_projection.flux_array = np.ones((10, 10))  # Dummy flux array
    mock_amoeba_disk.calculate_surface_intensity_map.return_value = mock_disk_projection

    mock_amoeba_map = MagicMock()
    mock_amoeba_convolution = MagicMock()
    # Make sure the mock convolution object has the attribute expected later
    mock_amoeba_convolution.magnification_array = (
        np.random.rand(50, 50) + 0.5
    )  # Use map size from fixture
    mock_amoeba_map.convolve_with_flux_projection.return_value = mock_amoeba_convolution

    # Patch util functions if necessary (or assume they work if simple)
    # Patching 'amoeba.Util.util.calculate_gravitational_radius' etc. might be needed
    # for full isolation, but let's focus on class interactions first.

    @pytest.mark.skipif(not AMOEBA_AVAILABLE, reason="amoeba package not installed")
    @patch(
        "slsim.Microlensing.light_curve.extract_light_curve",
        side_effect=mock_extract_light_curve,
    )
    @patch(
        "slsim.Microlensing.light_curve.AccretionDisk", return_value=mock_amoeba_disk
    )
    @patch(
        "slsim.Microlensing.light_curve.AmoebaMagnificationMap",
        return_value=mock_amoeba_map,
    )
    @patch("slsim.Microlensing.light_curve.util")  # Mock the whole util module
    def test_generate_agn_lightcurve_basic(
        self,
        mock_util,
        mock_AmoebaMap,
        mock_AccretionDisk,
        mock_extract,
        mlc_instance,
        cosmo,
    ):
        """Test basic AGN light curve generation (magnification)."""
        # Reset mocks that might persist between tests if defined outside
        mock_AccretionDisk.reset_mock()
        mock_AmoebaMap.reset_mock()
        mock_extract.reset_mock()
        mock_amoeba_disk.calculate_surface_intensity_map.reset_mock()
        mock_amoeba_map.convolve_with_flux_projection.reset_mock()

        # Set realistic return values for mock util functions if needed
        mock_util.calculate_gravitational_radius.return_value = 1.5e11  # meters
        mock_util.convert_cartesian_to_polar.return_value = (
            np.ones((10, 10)),
            np.ones((10, 10)),
        )  # dummy radii, phi
        mock_util.accretion_disk_temperature.return_value = (
            np.ones((10, 10)) * 5000
        )  # dummy temps

        source_redshift = 1.5
        deflector_redshift = 0.5
        v_transverse = 1200  # km/s
        smbh_mass_exp = 8.5
        wavelength = 500  # nm
        n_lc = 1

        lcs = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            lightcurve_type="magnification",
            v_transverse=v_transverse,
            num_lightcurves=n_lc,
            smbh_mass_exp=smbh_mass_exp,
            observer_frame_wavelength_in_nm=wavelength,
            # Other params use defaults
        )

        assert isinstance(lcs, list)  # Function returns list for AGN
        assert len(lcs) == n_lc
        assert isinstance(lcs[0], np.ndarray)
        assert lcs[0].shape == (MOCK_LC_LEN,)
        assert np.all(lcs[0] > 0)  # Magnification

        # Check AccretionDisk initialization
        mock_AccretionDisk.assert_called_once_with(
            smbh_mass_exp=smbh_mass_exp,
            redshift_source=source_redshift,
            inclination_angle=ANY,  # or check default 0
            corona_height=ANY,  # or check default 10
            temp_array=ANY,  # Check if it's the result of util.accretion_disk_temperature
            phi_array=ANY,  # Check if it's from util.convert_cartesian_to_polar
            g_array=ANY,  # Check if it's ones
            radii_array=ANY,  # Check if it's from util.convert_cartesian_to_polar
            height_array=ANY,  # Check if it's zeros
        )
        # Check surface intensity calculation call
        mock_amoeba_disk.calculate_surface_intensity_map.assert_called_once_with(
            observer_frame_wavelength_in_nm=wavelength
        )

        # Check AmoebaMagnificationMap initialization
        mock_AmoebaMap.assert_called_once_with(
            source_redshift,
            deflector_redshift,
            mlc_instance.magnification_map.magnifications,
            mlc_instance.magnification_map.kappa_tot,
            mlc_instance.magnification_map.shear,
            mean_microlens_mass_in_kg=ANY,  # check default or passed value
            total_microlens_einstein_radii=ANY,  # check calculation
            OmM=cosmo.Om0,
            H0=cosmo.H0.to(u.km / (u.s * u.Mpc)).value,
        )
        # Check convolution call
        mock_amoeba_map.convolve_with_flux_projection.assert_called_once_with(
            mock_disk_projection
        )

        # Check extract_light_curve call
        assert mock_extract.call_count == n_lc
        pixel_size_arcsec = mlc_instance.magnification_map.pixel_size
        kpc_per_arcsec = (
            cosmo.kpc_proper_per_arcmin(source_redshift).to(u.kpc / u.arcsec).value
            / 60.0
        )
        pixel_size_kpc = pixel_size_arcsec * kpc_per_arcsec
        pixel_size_meter = (pixel_size_kpc * u.kpc).to(u.m).value
        expected_time_years = mlc_instance.time_duration / 365.25

        mock_extract.assert_called_with(
            convolution_array=mock_amoeba_convolution.magnification_array,  # Check the result of amoeba convolution is passed
            pixel_size=pixel_size_meter,
            effective_transverse_velocity=v_transverse,
            light_curve_time_in_years=expected_time_years,
            pixel_shift=0,
            x_start_position=None,
            y_start_position=None,
            phi_travel_direction=None,
            return_track_coords=True,  # Called internally with True
            random_seed=None,
        )
        # Check that the internal convolved_map was updated
        assert hasattr(mlc_instance, "convolved_map")
        np.testing.assert_array_equal(
            mlc_instance.convolved_map, mock_amoeba_convolution.magnification_array
        )

    @pytest.mark.skipif(not AMOEBA_AVAILABLE, reason="amoeba package not installed")
    @patch(
        "slsim.Microlensing.light_curve.extract_light_curve",
        side_effect=mock_extract_light_curve,
    )
    @patch(
        "slsim.Microlensing.light_curve.AccretionDisk", return_value=mock_amoeba_disk
    )
    @patch(
        "slsim.Microlensing.light_curve.AmoebaMagnificationMap",
        return_value=mock_amoeba_map,
    )
    @patch("slsim.Microlensing.light_curve.util")  # Mock the whole util module
    def test_generate_agn_lightcurve_magnitude(
        self,
        mock_util,
        mock_AmoebaMap,
        mock_AccretionDisk,
        mock_extract,
        mlc_instance,
        cosmo,
    ):
        """Test AGN light curve generation (magnitude)."""
        # Reset mocks
        mock_AccretionDisk.reset_mock()
        mock_AmoebaMap.reset_mock()
        mock_extract.reset_mock()
        mock_amoeba_disk.calculate_surface_intensity_map.reset_mock()
        mock_amoeba_map.convolve_with_flux_projection.reset_mock()
        # Set return values for mocks if needed
        mock_util.calculate_gravitational_radius.return_value = 1.5e11
        mock_util.convert_cartesian_to_polar.return_value = (
            np.ones((10, 10)),
            np.ones((10, 10)),
        )
        mock_util.accretion_disk_temperature.return_value = np.ones((10, 10)) * 5000

        source_redshift = 1.5
        deflector_redshift = 0.5
        n_lc = 2

        lcs = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            lightcurve_type="magnitude",
            num_lightcurves=n_lc,
            # Use defaults for other params
        )

        assert isinstance(lcs, list)
        assert len(lcs) == n_lc
        assert isinstance(lcs[0], np.ndarray)
        assert lcs[0].shape == (MOCK_LC_LEN,)
        assert mock_extract.call_count == n_lc
        # Check magnitude calculation logic (similar to point source test)
        raw_mock_output = mock_extract_light_curve()[0]
        mean_convolved = np.nanmean(
            mlc_instance.convolved_map
        )  # Should be from mock_amoeba_convolution
        expected_mag_val = -2.5 * np.log10(raw_mock_output[0] / np.abs(mean_convolved))
        assert not np.allclose(
            lcs[0][0], raw_mock_output[0]
        )  # Check transformation occurred

    @pytest.mark.skipif(not AMOEBA_AVAILABLE, reason="amoeba package not installed")
    @patch(
        "slsim.Microlensing.light_curve.extract_light_curve",
        side_effect=mock_extract_light_curve,
    )
    @patch(
        "slsim.Microlensing.light_curve.AccretionDisk", return_value=mock_amoeba_disk
    )
    @patch(
        "slsim.Microlensing.light_curve.AmoebaMagnificationMap",
        return_value=mock_amoeba_map,
    )
    @patch("slsim.Microlensing.light_curve.util")  # Mock the whole util module
    def test_generate_agn_returns(
        self,
        mock_util,
        mock_AmoebaMap,
        mock_AccretionDisk,
        mock_extract,
        mlc_instance,
        cosmo,
    ):
        """Test return options for AGN light curves."""
        mock_AccretionDisk.reset_mock()
        mock_AmoebaMap.reset_mock()
        mock_extract.reset_mock()
        mock_util.calculate_gravitational_radius.return_value = 1.5e11
        mock_util.convert_cartesian_to_polar.return_value = (
            np.ones((10, 10)),
            np.ones((10, 10)),
        )
        mock_util.accretion_disk_temperature.return_value = np.ones((10, 10)) * 5000

        source_redshift = 1.5
        deflector_redshift = 0.5
        n_lc = 2

        # Test return tracks
        lcs, tracks = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            num_lightcurves=n_lc,
            return_track_coords=True,
            return_time_array=False,
        )
        assert isinstance(lcs, list) and len(lcs) == n_lc
        assert isinstance(tracks, list) and len(tracks) == n_lc
        assert isinstance(tracks[0], np.ndarray) and tracks[0].shape == (
            2,
            MOCK_LC_LEN,
        )  # (x,y), time_steps
        assert mock_extract.call_count == n_lc

        mock_extract.reset_mock()

        # Test return time
        lcs, times = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            num_lightcurves=n_lc,
            return_track_coords=False,
            return_time_array=True,
        )
        assert isinstance(lcs, list) and len(lcs) == n_lc
        assert isinstance(times, list) and len(times) == n_lc
        assert isinstance(times[0], np.ndarray) and len(times[0]) == MOCK_LC_LEN
        assert np.isclose(times[0][0], 0) and np.isclose(
            times[0][-1], mlc_instance.time_duration
        )
        assert mock_extract.call_count == n_lc

        mock_extract.reset_mock()

        # Test return tracks and time
        lcs, tracks, times = mlc_instance.generate_agn_lightcurve(
            source_redshift=source_redshift,
            deflector_redshift=deflector_redshift,
            cosmology=cosmo,
            num_lightcurves=n_lc,
            return_track_coords=True,
            return_time_array=True,
        )
        assert isinstance(lcs, list) and len(lcs) == n_lc
        assert isinstance(tracks, list) and len(tracks) == n_lc
        assert isinstance(times, list) and len(times) == n_lc
        assert tracks[0].shape == (2, MOCK_LC_LEN)
        assert len(times[0]) == MOCK_LC_LEN
        assert mock_extract.call_count == n_lc

    @patch(
        "slsim.Microlensing.light_curve.AMOEBA_AVAILABLE", False
    )  # Force unavailable
    def test_generate_agn_import_error(self, mlc_instance, cosmo):
        """Test that ImportError is raised if amoeba is unavailable."""
        source_redshift = 1.5
        deflector_redshift = 0.5

        with pytest.raises(ImportError, match="amoeba package is required"):
            mlc_instance.generate_agn_lightcurve(
                source_redshift=source_redshift,
                deflector_redshift=deflector_redshift,
                cosmology=cosmo,
            )
