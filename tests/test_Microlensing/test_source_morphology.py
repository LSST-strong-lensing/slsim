import pytest
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from unittest.mock import patch

from slsim.Microlensing.source_morphology.source_morphology import SourceMorphology
from slsim.Microlensing.source_morphology.gaussian import GaussianSourceMorphology
from slsim.Microlensing.source_morphology.agn import AGNSourceMorphology
from slsim.Microlensing.source_morphology.supernovae import SupernovaeSourceMorphology

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

try:
    import sncosmo

    SNCOSMO_AVAILABLE = True
except ImportError:
    SNCOSMO_AVAILABLE = False


# ---- Fixtures ----


@pytest.fixture
def cosmology():
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def kwargs_Gaussian(cosmology):
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "source_size": 0.05,
        "length_x": 1,
        "length_y": 1,
        "num_pix_x": 100,
        "num_pix_y": 100,
    }


@pytest.fixture
def kwargs_Gaussian_centered(cosmology):
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "source_size": 0.05,
        "length_x": 1,
        "length_y": 1,
        "num_pix_x": 101,
        "num_pix_y": 101,
        "center_x": 0.1,
        "center_y": -0.1,
    }


@pytest.fixture
def kwargs_AGN_band(cosmology):
    return {
        "source_redshift": 0.1,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "black_hole_mass_exponent": 8.0,
        "inclination_angle": 0,
        "black_hole_spin": 0,
        "observing_wavelength_band": "r",
        "eddington_ratio": 0.15,
    }


@pytest.fixture
def kwargs_AGN_wave(cosmology):
    return {
        "source_redshift": 0.1,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "black_hole_mass_exponent": 8.0,
        "inclination_angle": 30,
        "black_hole_spin": 0.5,
        "observer_frame_wavelength_in_nm": 600,
        "eddington_ratio": 0.15,
    }


@pytest.fixture
def gaussian_source(kwargs_Gaussian):
    return GaussianSourceMorphology(**kwargs_Gaussian)


@pytest.fixture
def gaussian_source_centered(kwargs_Gaussian_centered):
    return GaussianSourceMorphology(**kwargs_Gaussian_centered)


@pytest.fixture()
def agn_source_band(kwargs_AGN_band):
    if not ASTRO_UTIL_AVAILABLE or not SPECLITE_AVAILABLE:
        pytest.skip("Requires slsim.Util.astro_util and speclite")
    return AGNSourceMorphology(**kwargs_AGN_band)


@pytest.fixture()
def agn_source_wave(kwargs_AGN_wave):
    if not ASTRO_UTIL_AVAILABLE:
        pytest.skip("Requires slsim.Util.astro_util")
    return AGNSourceMorphology(**kwargs_AGN_wave)


def _make_user_snapshots(n=3, shape=(10, 10)):
    """Helper to create a minimal valid user_snapshots dict."""
    times = np.linspace(0, 20, n)
    kernels = [np.ones(shape) / np.prod(shape) for _ in range(n)]
    pixel_scales_m = np.linspace(1e13, 3e13, n)
    return {"times": times, "kernels": kernels, "pixel_scales_m": pixel_scales_m}


# ---- SourceMorphology Base Class Tests ----


class TestSourceMorphology:
    """Tests the SourceMorphology base class."""

    def test_base_class_init_static(self):
        base = SourceMorphology()
        assert isinstance(base, SourceMorphology)
        assert base.is_time_varying is False
        assert base.user_snapshots is None

    def test_base_class_init_time_varying_flag(self):
        base = SourceMorphology(is_time_varying=True)
        assert base.is_time_varying is True

    def test_base_class_init_with_user_snapshots(self):
        """Providing user_snapshots should set is_time_varying=True and call
        _prepare_snapshots."""
        snapshots = _make_user_snapshots(n=3)
        base = SourceMorphology(user_snapshots=snapshots)
        assert base.is_time_varying is True
        assert hasattr(base, "_anchor_times")
        assert hasattr(base, "_anchor_kernels_3d")
        assert hasattr(base, "_anchor_scales")

    def test_prepare_snapshots_sorts_by_time(self):
        """Snapshots provided out of order should be sorted by time."""
        times_unsorted = np.array([10.0, 0.0, 5.0])
        kernels = [np.eye(5) * i for i in range(3)]
        pixel_scales_m = np.array([3e13, 1e13, 2e13])
        snapshots = {
            "times": times_unsorted,
            "kernels": kernels,
            "pixel_scales_m": pixel_scales_m,
        }
        base = SourceMorphology(user_snapshots=snapshots)
        assert base._anchor_times[0] < base._anchor_times[1] < base._anchor_times[2]
        np.testing.assert_array_equal(base._anchor_times, np.sort(times_unsorted))

    def test_prepare_snapshots_pads_kernels_to_same_shape(self):
        """Kernels of different sizes should be padded to the max shape."""
        times = np.array([0.0, 10.0, 20.0])
        kernels = [
            np.ones((5, 5)) / 25,
            np.ones((8, 8)) / 64,
            np.ones((6, 7)) / 42,
        ]
        pixel_scales_m = np.array([1e13, 2e13, 3e13])
        snapshots = {
            "times": times,
            "kernels": kernels,
            "pixel_scales_m": pixel_scales_m,
        }
        base = SourceMorphology(user_snapshots=snapshots)
        # All stacked kernels should have shape (max_y=8, max_x=8)
        assert base._anchor_kernels_3d.shape == (3, 8, 8)

    def test_interpolate_snapshots_returns_correct_types(self):
        """_interpolate_snapshots should return lists of arrays and floats."""
        snapshots = _make_user_snapshots(n=5)
        base = SourceMorphology(user_snapshots=snapshots)
        requested = np.array([2.0, 8.0, 15.0])
        kernels, scales = base._interpolate_snapshots(requested)
        assert isinstance(kernels, list)
        assert isinstance(scales, list)
        assert len(kernels) == 3
        assert len(scales) == 3
        for k in kernels:
            assert isinstance(k, np.ndarray)

    def test_interpolate_snapshots_clamps_out_of_bounds(self):
        """Times outside the anchor range should be clamped (not
        extrapolated)."""
        snapshots = _make_user_snapshots(n=3)
        base = SourceMorphology(user_snapshots=snapshots)
        # Request times way outside the anchor range
        requested = np.array([-100.0, 1000.0])
        kernels, scales = base._interpolate_snapshots(requested)
        assert len(kernels) == 2
        # Should not raise or return NaN
        for k in kernels:
            assert np.all(np.isfinite(k))

    def test_interpolate_snapshots_normalization(self):
        """Interpolated kernels should sum to 1."""
        snapshots = _make_user_snapshots(n=4)
        base = SourceMorphology(user_snapshots=snapshots)
        requested = np.linspace(0, 20, 10)
        kernels, _ = base._interpolate_snapshots(requested)
        for k in kernels:
            np.testing.assert_allclose(np.nansum(k), 1.0, rtol=1e-5)

    def test_get_time_dependent_kernel_maps_with_snapshots(self):
        """get_time_dependent_kernel_maps routes through _interpolate_snapshots
        when user_snapshots are provided."""
        snapshots = _make_user_snapshots(n=3)
        base = SourceMorphology(user_snapshots=snapshots)
        time_anchors = np.array([1.0, 5.0, 10.0])
        kernels, scales = base.get_time_dependent_kernel_maps(time_anchors)
        assert len(kernels) == 3
        assert len(scales) == 3

    def test_get_kernel_map_not_implemented(self):
        base = SourceMorphology()
        with pytest.raises(NotImplementedError):
            base.get_kernel_map()
        with pytest.raises(NotImplementedError):
            _ = base.kernel_map

    def test_kernel_map_raises_for_time_varying(self):
        """Accessing kernel_map on a time-varying source should raise
        AttributeError."""
        snapshots = _make_user_snapshots(n=3)
        base = SourceMorphology(user_snapshots=snapshots)
        with pytest.raises(
            AttributeError,
            match="Time-varying sources do not have a single static kernel_map",
        ):
            _ = base.kernel_map

    def test_base_class_pixel_scale_attribute_error(self):
        base = SourceMorphology()
        with pytest.raises(AttributeError):
            _ = base.length_x
        with pytest.raises(AttributeError):
            _ = base.pixel_scale_x
        with pytest.raises(AttributeError, match="Pixel scale not defined."):
            _ = base.pixel_scale
        with pytest.raises(AttributeError):
            _ = base.pixel_scale_x_m
        with pytest.raises(AttributeError):
            _ = base.pixel_scale_y_m
        with pytest.raises(AttributeError):
            _ = base.pixel_scale_m

    def test_conversions(self, gaussian_source, cosmology):
        arcsecs_val = 1.0
        redshift = gaussian_source.source_redshift
        metres_val = gaussian_source.arcsecs_to_metres(arcsecs_val, cosmology, redshift)
        assert isinstance(metres_val, float) and metres_val > 0
        arcsecs_round_trip = gaussian_source.metres_to_arcsecs(
            metres_val, cosmology, redshift
        )
        np.testing.assert_allclose(arcsecs_round_trip, arcsecs_val, rtol=1e-6)
        assert gaussian_source.arcsecs_to_metres(0, cosmology, redshift) == 0.0
        assert gaussian_source.metres_to_arcsecs(0, cosmology, redshift) == 0.0

    def test_property_caching_gaussian(self, gaussian_source):
        """Tests that kernel_map and pixel_scale are cached after first
        access."""
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
            assert np.array_equal(kernel1, initial_kernel)

    def test_pixel_scale_caching(self, gaussian_source):
        """pixel_scale should be cached after first access."""
        assert hasattr(gaussian_source, "_pixel_scale")
        initial_pixel_scale = gaussian_source._pixel_scale
        del gaussian_source._pixel_scale
        scale1 = gaussian_source.pixel_scale
        assert hasattr(gaussian_source, "_pixel_scale")
        assert scale1 == initial_pixel_scale
        scale2 = gaussian_source.pixel_scale
        assert scale1 == scale2


# ---- GaussianSourceMorphology Tests ----


class TestGaussianSourceMorphology:

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
        assert gaussian_source.is_time_varying is False

    def test_get_kernel_map_shape_and_normalization(
        self, gaussian_source, kwargs_Gaussian
    ):
        kernel = gaussian_source.get_kernel_map()
        assert kernel.shape == (
            kwargs_Gaussian["num_pix_y"],
            kwargs_Gaussian["num_pix_x"],
        )
        np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-6)

    def test_get_kernel_map_peak_at_center(self, gaussian_source, kwargs_Gaussian):
        kernel = gaussian_source.kernel_map
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

    def test_pixel_scale_properties(self, gaussian_source, kwargs_Gaussian):
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

    def test_pixel_scale_in_metres(self, gaussian_source):
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

    def test_get_time_dependent_kernel_maps_static_fallback(self, gaussian_source):
        """For a static Gaussian, get_time_dependent_kernel_maps should
        replicate the kernel for each requested time."""
        times = np.array([0.0, 10.0, 50.0])
        kernels, scales = gaussian_source.get_time_dependent_kernel_maps(times)
        assert len(kernels) == 3
        assert len(scales) == 3
        for k in kernels:
            np.testing.assert_array_equal(k, gaussian_source.kernel_map)
        for s in scales:
            assert s == gaussian_source.pixel_scale_m


# ---- AGNSourceMorphology Tests ----


@pytest.mark.skipif(not ASTRO_UTIL_AVAILABLE, reason="slsim.Util.astro_util not found")
class TestAGNSourceMorphology:
    """Tests the AGNSourceMorphology class using real dependencies."""

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
        assert agn_source_band.is_time_varying is False

        filter_r = speclite.filters.load_filter("lsst2023-r")
        expected_wave_nm = filter_r.effective_wavelength.to(u.nm).value
        np.testing.assert_allclose(
            agn_source_band.observer_frame_wavelength_in_nm, expected_wave_nm
        )

        # ---------------------------------------------------------------
        # Pixel-scale metadata is now computed analytically in __init__,
        # so these assertions hold WITHOUT having triggered get_kernel_map().
        # ---------------------------------------------------------------
        assert agn_source_band.pixel_scale_x_m > 0
        assert agn_source_band.pixel_scale_y_m > 0
        assert agn_source_band.pixel_scale_m > 0
        assert agn_source_band.pixel_scale_x > 0
        assert agn_source_band.pixel_scale_y > 0
        assert agn_source_band.pixel_scale > 0
        assert agn_source_band.length_x > 0
        assert agn_source_band.length_y > 0

        # num_pix is set analytically to 2 * r_resolution
        expected_num_pix = 2 * kwargs_AGN_band["r_resolution"]
        assert agn_source_band.num_pix_x == expected_num_pix
        assert agn_source_band.num_pix_y == expected_num_pix

        # Trigger actual map computation and verify it has the expected shape.
        kernel = agn_source_band.kernel_map
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (expected_num_pix, expected_num_pix)

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
        assert agn_source_wave.is_time_varying is False

        expected_num_pix = 2 * kwargs_AGN_wave["r_resolution"]
        assert agn_source_wave.num_pix_x == expected_num_pix
        assert agn_source_wave.num_pix_y == expected_num_pix
        assert agn_source_wave.pixel_scale_x_m > 0
        assert agn_source_wave.length_x > 0

        # Trigger map computation.
        kernel = agn_source_wave.kernel_map
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (expected_num_pix, expected_num_pix)

    def test_get_kernel_map_normalization(self, agn_source_wave):
        """Tests the generated AGN kernel map properties."""
        kernel = agn_source_wave.kernel_map
        np.testing.assert_allclose(np.nansum(kernel), 1.0, rtol=1e-5)
        assert np.all(kernel >= 0)

    def test_pixel_scales_are_consistent(self, agn_source_wave, kwargs_AGN_wave):
        """Pixel scale derived analytically must be consistent with
        gravitational-radius calculation."""
        grav_radius = calculate_gravitational_radius(
            kwargs_AGN_wave["black_hole_mass_exponent"]
        )
        grav_radius_m = grav_radius.to(u.m).value

        num_pix = 2 * kwargs_AGN_wave["r_resolution"]
        expected_pixel_scale_m = 2 * kwargs_AGN_wave["r_out"] * grav_radius_m / num_pix

        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_x_m, expected_pixel_scale_m, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_y_m, expected_pixel_scale_m, rtol=1e-6
        )
        np.testing.assert_allclose(
            agn_source_wave.pixel_scale_m, expected_pixel_scale_m, rtol=1e-6
        )

    def test_arcsec_scales_derived_from_metre_scales(self, agn_source_wave):
        """Arcsecond pixel scales must be the metre scales converted via the
        cosmology."""
        expected_x = agn_source_wave.metres_to_arcsecs(
            agn_source_wave.pixel_scale_x_m,
            agn_source_wave.cosmo,
            agn_source_wave.source_redshift,
        )
        expected_y = agn_source_wave.metres_to_arcsecs(
            agn_source_wave.pixel_scale_y_m,
            agn_source_wave.cosmo,
            agn_source_wave.source_redshift,
        )
        np.testing.assert_allclose(agn_source_wave.pixel_scale_x, expected_x, rtol=1e-6)
        np.testing.assert_allclose(agn_source_wave.pixel_scale_y, expected_y, rtol=1e-6)

    def test_length_equals_numpix_times_pixscale(self, agn_source_wave):
        """length_x/y must equal num_pix * pixel_scale in arcseconds."""
        np.testing.assert_allclose(
            agn_source_wave.length_x,
            agn_source_wave.num_pix_x * agn_source_wave.pixel_scale_x,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            agn_source_wave.length_y,
            agn_source_wave.num_pix_y * agn_source_wave.pixel_scale_y,
            rtol=1e-6,
        )

    def test_pixel_scales_set_before_kernel_map_access(self, kwargs_AGN_wave):
        """Pixel scales must be available immediately after __init__, before
        kernel_map is ever accessed.

        This verifies the analytical derivation does not depend on the
        2-D map.
        """
        agn = AGNSourceMorphology(**kwargs_AGN_wave)
        # Ensure the kernel has NOT been computed yet.
        assert not hasattr(agn, "_kernel_map")
        # All scale attributes should still be set.
        assert agn.pixel_scale_x_m > 0
        assert agn.pixel_scale_y_m > 0
        assert agn.pixel_scale_m > 0
        assert agn.pixel_scale_x > 0
        assert agn.pixel_scale_y > 0
        assert agn.num_pix_x == 2 * kwargs_AGN_wave["r_resolution"]
        assert agn.num_pix_y == 2 * kwargs_AGN_wave["r_resolution"]

    def test_static_agn_is_not_time_varying(self, agn_source_wave):
        assert agn_source_wave.is_time_varying is False

    def test_time_varying_agn_with_user_snapshots(self, kwargs_AGN_wave):
        """AGN constructed with user_snapshots should be time-varying and
        return interpolated kernels via get_time_dependent_kernel_maps()."""
        snapshots = _make_user_snapshots(n=4)
        agn = AGNSourceMorphology(
            **kwargs_AGN_wave,
            user_snapshots=snapshots,
        )
        assert agn.is_time_varying is True

        # kernel_map property must raise for time-varying sources.
        with pytest.raises(AttributeError, match="Time-varying"):
            _ = agn.kernel_map

        # get_time_dependent_kernel_maps must work and return the right count.
        times = np.array([0.0, 5.0, 15.0])
        kernels, scales = agn.get_time_dependent_kernel_maps(times)
        assert len(kernels) == 3
        assert len(scales) == 3
        for k in kernels:
            assert isinstance(k, np.ndarray)
            assert np.all(np.isfinite(k))
            np.testing.assert_allclose(np.nansum(k), 1.0, rtol=1e-5)

    def test_time_varying_agn_representative_pixel_scale(self, kwargs_AGN_wave):
        """For a time-varying AGN, the representative pixel scale should come
        from the first entry of user_snapshots['pixel_scales_m']."""
        snapshots = _make_user_snapshots(n=3)
        agn = AGNSourceMorphology(**kwargs_AGN_wave, user_snapshots=snapshots)
        expected_rep_scale = float(snapshots["pixel_scales_m"][0])
        assert agn.pixel_scale_x_m == expected_rep_scale
        assert agn.pixel_scale_y_m == expected_rep_scale
        assert agn.pixel_scale_m == expected_rep_scale

    def test_time_varying_agn_num_pix_from_first_snapshot(self, kwargs_AGN_wave):
        """num_pix_x/y for a time-varying AGN should come from the first
        kernel's shape."""
        n_pix = 15
        snapshots = _make_user_snapshots(n=3, shape=(n_pix, n_pix))
        agn = AGNSourceMorphology(**kwargs_AGN_wave, user_snapshots=snapshots)
        assert agn.num_pix_x == n_pix
        assert agn.num_pix_y == n_pix

    def test_time_varying_agn_without_snapshots_raises(self, kwargs_AGN_wave):
        """AGN with is_time_varying=True but no user_snapshots should raise
        NotImplementedError."""
        with pytest.raises(NotImplementedError, match="currently only supported via"):
            AGNSourceMorphology(**kwargs_AGN_wave, is_time_varying=True)

    def test_build_analytical_snapshots_raises(self, kwargs_AGN_wave):
        """Directly testing the internal helper method for analytical
        snapshots."""
        agn = AGNSourceMorphology(**kwargs_AGN_wave)
        with pytest.raises(NotImplementedError, match="currently only supported via"):
            agn._build_analytical_snapshots()

    def test_no_more_get_variable_kernel_map(self, agn_source_wave):
        """Confirm that the removed placeholder methods no longer exist."""
        assert not hasattr(agn_source_wave, "get_variable_kernel_map")
        assert not hasattr(agn_source_wave, "get_integrated_kernel_map")


# ---- SupernovaeSourceMorphology Tests ----


@pytest.mark.skipif(not SNCOSMO_AVAILABLE, reason="sncosmo not installed")
class TestSupernovaeSourceMorphology:
    """Tests the new SupernovaeSourceMorphology class."""

    @pytest.fixture
    def cosmo(self):
        return FlatLambdaCDM(H0=70, Om0=0.3)

    @pytest.fixture
    def sn_source_r(self, cosmo):
        """A basic supernovae source in the r-band (uses sncosmo)."""
        return SupernovaeSourceMorphology(
            observing_wavelength_band="r",
            source_redshift=0.5,
            cosmo=cosmo,
            grid_pixels=50,  # small for speed in tests
            anchor_spacing_days=20.0,  # coarse for speed
        )

    @pytest.fixture
    def sn_source_user_snapshots(self, cosmo):
        """A supernovae source with user-provided snapshots (bypasses sncosmo
        full calculation)."""
        snapshots = _make_user_snapshots(n=4, shape=(50, 50))
        return SupernovaeSourceMorphology(
            observing_wavelength_band="r",
            source_redshift=0.5,
            cosmo=cosmo,
            user_snapshots=snapshots,
        )

    def test_init_is_time_varying(self, sn_source_r):
        """SupernovaeSourceMorphology should always be time-varying."""
        assert sn_source_r.is_time_varying is True

    def test_init_band_conversion(self, cosmo):
        """Band conversion from LSST band name to sncosmo format."""
        sn = SupernovaeSourceMorphology(
            observing_wavelength_band="i",
            source_redshift=0.5,
            cosmo=cosmo,
            grid_pixels=50,
            anchor_spacing_days=20.0,
        )
        # sncosmo_fmt for LSST is lambda band: f"lsst{band}"
        assert sn.band == "lssti"

    def test_init_custom_band_passthrough(self, cosmo):
        """A non-registered band name should be passed through directly."""
        # "bessellb" is a valid sncosmo filter but not a registered SLSim band
        sn = SupernovaeSourceMorphology(
            observing_wavelength_band="bessellb",
            source_redshift=0.5,
            cosmo=cosmo,
            grid_pixels=50,
            anchor_spacing_days=20.0,
        )
        assert sn.band == "bessellb"

    def test_kernel_map_raises_for_time_varying(self, sn_source_r):
        """kernel_map property should raise AttributeError for SN (time-
        varying)."""
        with pytest.raises(AttributeError, match="Time-varying"):
            _ = sn_source_r.kernel_map

    def test_get_kernel_map_returns_normalized_array(self, sn_source_r):
        """get_kernel_map(time_days) should return a normalized 2D array and a
        pixel scale."""
        kernel, pixel_scale_m = sn_source_r.get_kernel_map(time_days=5.0)
        assert isinstance(kernel, np.ndarray)
        assert kernel.ndim == 2
        assert kernel.shape == (sn_source_r._num_pix_y, sn_source_r._num_pix_x)
        np.testing.assert_allclose(np.nansum(kernel), 1.0, rtol=1e-5)
        assert isinstance(pixel_scale_m, float)
        assert pixel_scale_m > 0

    def test_get_kernel_map_grows_with_time(self, sn_source_r):
        """The physical pixel scale (and thus source size) should grow with
        time."""
        _, scale_early = sn_source_r.get_kernel_map(time_days=1.0)
        _, scale_late = sn_source_r.get_kernel_map(time_days=10.0)
        assert scale_late > scale_early

    def test_get_time_dependent_kernel_maps(self, sn_source_r):
        """get_time_dependent_kernel_maps should return interpolated kernels
        via user_snapshots."""
        times = np.array([0.0, 5.0, 10.0])
        kernels, scales = sn_source_r.get_time_dependent_kernel_maps(times)
        assert len(kernels) == 3
        assert len(scales) == 3
        for k in kernels:
            assert isinstance(k, np.ndarray)
            assert np.all(np.isfinite(k))
            np.testing.assert_allclose(np.nansum(k), 1.0, rtol=1e-5)

    def test_user_snapshots_bypass(self, sn_source_user_snapshots):
        """With user_snapshots provided, sncosmo calculations are bypassed."""
        assert sn_source_user_snapshots.is_time_varying is True
        times = np.array([2.0, 10.0, 18.0])
        kernels, scales = sn_source_user_snapshots.get_time_dependent_kernel_maps(times)
        assert len(kernels) == 3

    def test_invalid_band_raises_value_error(self, cosmo):
        """An unrecognized sncosmo band should raise ValueError."""
        with pytest.raises(ValueError, match="not recognized by sncosmo"):
            SupernovaeSourceMorphology(
                observing_wavelength_band="totally_invalid_band_xyz",
                source_redshift=0.5,
                cosmo=cosmo,
                grid_pixels=50,
                anchor_spacing_days=20.0,
            )

    def test_ellipticity_affects_kernel(self, cosmo):
        """Non-unit ellipticity should produce a different kernel than
        spherical."""
        sn_spherical = SupernovaeSourceMorphology(
            observing_wavelength_band="r",
            source_redshift=0.5,
            cosmo=cosmo,
            ellipticity=1.0,
            grid_pixels=50,
            anchor_spacing_days=20.0,
        )
        sn_elliptical = SupernovaeSourceMorphology(
            observing_wavelength_band="r",
            source_redshift=0.5,
            cosmo=cosmo,
            ellipticity=0.7,
            grid_pixels=50,
            anchor_spacing_days=20.0,
        )
        k_sph, _ = sn_spherical.get_kernel_map(time_days=5.0)
        k_ell, _ = sn_elliptical.get_kernel_map(time_days=5.0)
        assert not np.allclose(k_sph, k_ell)

    def test_continuous_monochromatic_morphology_zero_mask(self, sn_source_r):
        """When all radii exceed r_phot, intensity should be zero
        everywhere."""
        # Very early time (nearly t=0) means r_phot is tiny (clamped to 1e8 m)
        r_large = np.linspace(1e10, 1e12, 100)  # all beyond r_phot
        # Since r_phot = max(v * t, 1e8) ~ 1e8 at t=0, and all r_large >> 1e8
        intensity = sn_source_r._continuous_monochromatic_morphology(
            wavelength_angstroms=5000, time_seconds=0.0, R_eff=r_large
        )
        assert np.all(intensity == 0.0)
