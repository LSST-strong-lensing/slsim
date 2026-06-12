import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from slsim.Microlensing.lightcurve import MicrolensingLightCurve
from slsim.Microlensing.magmap import MagnificationMap
from slsim.Microlensing.source_morphology.gaussian import GaussianSourceMorphology
from slsim.Microlensing.source_morphology.agn import AGNSourceMorphology

# ---- Fixtures ----


@pytest.fixture
def cosmology():
    return FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def theta_star():
    return 4e-6  # arcsec


@pytest.fixture
def magmap_instance(theta_star):
    try:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        magmap2D_path = os.path.join(
            test_dir, "..", "TestData", "test_magmaps_microlensing", "magmap_0.npy"
        )
        magmap2D = np.load(magmap2D_path)
    except Exception as e:
        pytest.fail(
            f"Failed to load TestData/test_magmaps_microlensing/magmap_0.npy: {e}"
        )

    kwargs_MagnificationMap = {
        "kappa_tot": 0.47128266,
        "shear": 0.42394672,
        "kappa_star": 0.12007537,
        "theta_star": theta_star,
        "center_x": 0.0,
        "center_y": 0.0,
        "half_length_x": 2.5 * theta_star,
        "half_length_y": 2.5 * theta_star,
        "mass_function": "kroupa",
        "m_solar": 1.0,
        "m_lower": 0.01,
        "m_upper": 5,
        "num_pixels_x": 50,
        "num_pixels_y": 50,
        "kwargs_IPM": {},
    }
    return MagnificationMap(magnifications_array=magmap2D, **kwargs_MagnificationMap)


@pytest.fixture
def observation_time_array():
    """A simple observation time array spanning 4000 days."""
    return np.linspace(0, 4000, 40)


@pytest.fixture
def kwargs_source_morphology_Gaussian(cosmology):
    return {"source_redshift": 0.5, "cosmo": cosmology, "source_size": 1e-7}


@pytest.fixture
def kwargs_source_morphology_AGN_wave(cosmology):
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "black_hole_mass_exponent": 8,
        "inclination_angle": 30,
        "black_hole_spin": 0,
        "observer_frame_wavelength_in_nm": 600,
        "eddington_ratio": 0.1,
    }


@pytest.fixture
def kwargs_source_morphology_AGN_band(cosmology):
    return {
        "source_redshift": 0.5,
        "cosmo": cosmology,
        "r_out": 1000,
        "r_resolution": 100,
        "black_hole_mass_exponent": 8,
        "inclination_angle": 0,
        "black_hole_spin": 0,
        "observing_wavelength_band": "r",
        "eddington_ratio": 0.1,
    }


@pytest.fixture
def ml_lc_gaussian(
    magmap_instance, observation_time_array, kwargs_source_morphology_Gaussian
):
    return MicrolensingLightCurve(
        magnification_map=magmap_instance,
        observation_time_array=observation_time_array,
        point_source_morphology="gaussian",
        kwargs_source_morphology=kwargs_source_morphology_Gaussian,
    )


@pytest.fixture
def ml_lc_agn_wave(
    magmap_instance, observation_time_array, kwargs_source_morphology_AGN_wave
):
    return MicrolensingLightCurve(
        magnification_map=magmap_instance,
        observation_time_array=observation_time_array,
        point_source_morphology="agn",
        kwargs_source_morphology=kwargs_source_morphology_AGN_wave,
    )


@pytest.fixture
def ml_lc_agn_band(
    magmap_instance, observation_time_array, kwargs_source_morphology_AGN_band
):
    return MicrolensingLightCurve(
        magnification_map=magmap_instance,
        observation_time_array=observation_time_array,
        point_source_morphology="agn",
        kwargs_source_morphology=kwargs_source_morphology_AGN_band,
    )


# ---- Tests ----


class TestMicrolensingLightCurveInit:

    def test_init_gaussian(
        self, magmap_instance, observation_time_array, kwargs_source_morphology_Gaussian
    ):
        ml_lc = MicrolensingLightCurve(
            magnification_map=magmap_instance,
            observation_time_array=observation_time_array,
            point_source_morphology="gaussian",
            kwargs_source_morphology=kwargs_source_morphology_Gaussian,
        )
        assert ml_lc.magnification_map is magmap_instance
        assert ml_lc.time_duration_observer_frame == pytest.approx(
            observation_time_array[-1] - observation_time_array[0]
        )
        assert isinstance(ml_lc._source_morphology, GaussianSourceMorphology)

    def test_init_with_source_morphology_instance(
        self,
        magmap_instance,
        observation_time_array,
        kwargs_source_morphology_Gaussian,
        cosmology,
    ):
        """When a source_morphology_instance is passed, it should be used
        directly."""
        shared_morph = GaussianSourceMorphology(
            source_redshift=0.5,
            cosmo=cosmology,
            source_size=1e-7,
            length_x=magmap_instance.half_length_x * 2,
            length_y=magmap_instance.half_length_y * 2,
            num_pix_x=magmap_instance.num_pixels_x,
            num_pix_y=magmap_instance.num_pixels_y,
        )
        ml_lc = MicrolensingLightCurve(
            magnification_map=magmap_instance,
            observation_time_array=observation_time_array,
            point_source_morphology="gaussian",
            kwargs_source_morphology=kwargs_source_morphology_Gaussian,
            source_morphology_instance=shared_morph,
        )
        assert ml_lc._source_morphology is shared_morph

    def test_init_invalid_morphology_raises(
        self, magmap_instance, observation_time_array
    ):
        with pytest.raises(ValueError, match="Invalid source morphology type"):
            MicrolensingLightCurve(
                magnification_map=magmap_instance,
                observation_time_array=observation_time_array,
                point_source_morphology="invalid_type",
                kwargs_source_morphology={},
            )

    def test_init_supernovae_raises_not_implemented(
        self, magmap_instance, observation_time_array
    ):
        """SupernovaeSourceMorphology requires sncosmo; with no kwargs it
        raises."""
        with pytest.raises(Exception):
            MicrolensingLightCurve(
                magnification_map=magmap_instance,
                observation_time_array=observation_time_array,
                point_source_morphology="supernovae",
                kwargs_source_morphology={},
            )

    def test_properties(self, ml_lc_gaussian, magmap_instance, observation_time_array):
        assert ml_lc_gaussian.magnification_map is magmap_instance
        assert ml_lc_gaussian.time_duration_observer_frame == pytest.approx(
            observation_time_array[-1] - observation_time_array[0]
        )


class TestMicrolensingLightCurveGaussian:

    def test_generate_lightcurves_returns_correct_structure(
        self, ml_lc_gaussian, cosmology
    ):
        lcs, tracks, time_arrays = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            lightcurve_type="magnitude",
            num_lightcurves=2,
        )
        assert isinstance(lcs, list)
        assert len(lcs) == 2
        assert isinstance(tracks, list)
        assert len(tracks) == 2
        assert isinstance(time_arrays, list)
        assert len(time_arrays) == 2

    def test_generate_lightcurves_magnitude_output(self, ml_lc_gaussian, cosmology):
        lcs, _, time_arrays = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            lightcurve_type="magnitude",
            num_lightcurves=1,
        )
        lc = lcs[0]
        assert isinstance(lc, np.ndarray)
        assert len(lc) > 0
        assert np.issubdtype(lc.dtype, np.floating)
        assert not np.any(np.isnan(lc))
        assert not np.any(np.isinf(lc))

    def test_generate_lightcurves_magnification_output(self, ml_lc_gaussian, cosmology):
        lcs, _, _ = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            lightcurve_type="magnification",
            num_lightcurves=1,
        )
        lc = lcs[0]
        assert isinstance(lc, np.ndarray)
        assert np.all(lc >= 0)  # magnifications are non-negative
        assert not np.any(np.isnan(lc))

    def test_generate_lightcurves_invalid_type_raises(self, ml_lc_gaussian, cosmology):
        with pytest.raises(ValueError, match="Lightcurve type not recognized"):
            ml_lc_gaussian.generate_lightcurves(
                source_redshift=0.5,
                cosmo=cosmology,
                lightcurve_type="invalid_type",
                num_lightcurves=1,
            )

    def test_generate_lightcurves_time_arrays_match_observation(
        self, ml_lc_gaussian, cosmology, observation_time_array
    ):
        """The returned time arrays should match the observation time array."""
        _, _, time_arrays = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            num_lightcurves=1,
        )
        np.testing.assert_allclose(time_arrays[0], observation_time_array)

    def test_generate_lightcurves_lc_length_matches_observation_time(
        self, ml_lc_gaussian, cosmology, observation_time_array
    ):
        """The lightcurve length should match the number of observation time
        steps."""
        lcs, _, _ = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            num_lightcurves=1,
        )
        assert len(lcs[0]) == len(observation_time_array)

    def test_generate_lightcurves_with_specific_start_and_angle(
        self, ml_lc_gaussian, cosmology
    ):
        """Specifying start position and angle should not raise."""
        # half_x = ml_lc_gaussian.magnification_map.half_length_x
        # half_y = ml_lc_gaussian.magnification_map.half_length_y
        lcs, tracks, _ = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            num_lightcurves=1,
            x_start_position=0.0,
            y_start_position=0.0,
            phi_travel_direction=45.0,
        )
        assert isinstance(lcs, list) and len(lcs) == 1
        assert isinstance(lcs[0], np.ndarray) and len(lcs[0]) > 0

    def test_track_coords_shape(self, ml_lc_gaussian, cosmology):
        """Track coordinates should have shape (2, N) for x and y."""
        _, tracks, _ = ml_lc_gaussian.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            num_lightcurves=1,
            x_start_position=0.0,
            y_start_position=0.0,
            phi_travel_direction=0.0,
        )
        track = tracks[0]
        assert isinstance(track, np.ndarray)
        assert track.shape[0] == 2


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in divide:RuntimeWarning"
)
class TestMicrolensingLightCurveAGN:

    def test_generate_lightcurves_agn_wave(self, ml_lc_agn_wave, cosmology):
        lcs, _, _ = ml_lc_agn_wave.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            lightcurve_type="magnification",
            num_lightcurves=1,
        )
        assert isinstance(lcs, list) and len(lcs) == 1
        lc = lcs[0]
        assert isinstance(lc, np.ndarray) and len(lc) > 0
        assert not np.any(np.isnan(lc))

    def test_generate_lightcurves_agn_band(self, ml_lc_agn_band, cosmology):
        lcs, _, _ = ml_lc_agn_band.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            lightcurve_type="magnitude",
            num_lightcurves=1,
        )
        assert isinstance(lcs, list) and len(lcs) == 1
        lc = lcs[0]
        assert isinstance(lc, np.ndarray) and len(lc) > 0
        assert not np.any(np.isnan(lc))


class TestMicrolensingLightCurveTimeVarying:
    """Tests for time-varying source morphology path in
    MicrolensingLightCurve."""

    @pytest.fixture
    def user_snapshots_fixture(self):
        """A minimal valid user_snapshots dictionary."""
        n = 5
        shape = (50, 50)
        times = np.linspace(0, 40, n)
        kernels = [np.ones(shape) / float(np.prod(shape)) for _ in range(n)]
        pixel_scales_m = np.linspace(1e13, 5e13, n)
        return {
            "times": times,
            "kernels": kernels,
            "pixel_scales_m": pixel_scales_m,
        }

    @pytest.fixture
    def cosmology(self):
        return FlatLambdaCDM(H0=70, Om0=0.3)

    @pytest.fixture
    def ml_lc_time_varying(self, magmap_instance, user_snapshots_fixture, cosmology):
        """Build a MicrolensingLightCurve with a time-varying AGN morphology.

        AGNSourceMorphology now accepts user_snapshots correctly (pixel
        scales are derived analytically, not from kernel_map), so we can
        use it directly as the shared morphology instance.
        """
        kwargs_agn = {
            "source_redshift": 0.5,
            "cosmo": cosmology,
            "r_out": 1000,
            "r_resolution": 100,
            "black_hole_mass_exponent": 8,
            "inclination_angle": 0,
            "black_hole_spin": 0,
            "observer_frame_wavelength_in_nm": 600,
            "eddington_ratio": 0.1,
            "user_snapshots": user_snapshots_fixture,
        }
        morph_instance = AGNSourceMorphology(**kwargs_agn)

        obs_times = np.linspace(0, 40, 20)
        return MicrolensingLightCurve(
            magnification_map=magmap_instance,
            observation_time_array=obs_times,
            point_source_morphology="agn",
            kwargs_source_morphology=kwargs_agn,
            source_morphology_instance=morph_instance,
        )

    def test_time_varying_source_is_detected(self, ml_lc_time_varying):
        """The injected morphology must report is_time_varying=True."""
        assert ml_lc_time_varying._source_morphology.is_time_varying is True

    def test_time_varying_lightcurve_generation(self, ml_lc_time_varying, cosmology):
        """Time-varying morphology path should produce a valid lightcurve."""
        lcs, tracks, time_arrays = ml_lc_time_varying.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            lightcurve_type="magnitude",
            num_lightcurves=1,
        )
        assert len(lcs) == 1
        lc = lcs[0]
        assert isinstance(lc, np.ndarray)
        assert len(lc) > 0
        assert not np.any(np.isnan(lc))
        assert not np.any(np.isinf(lc))

    def test_time_varying_lc_length_matches_obs_times(
        self, ml_lc_time_varying, cosmology
    ):
        """Lightcurve length must match the observation_time_array length."""
        lcs, _, time_arrays = ml_lc_time_varying.generate_lightcurves(
            source_redshift=0.5,
            cosmo=cosmology,
            num_lightcurves=1,
        )
        assert len(lcs[0]) == 20
        np.testing.assert_allclose(time_arrays[0], np.linspace(0, 40, 20))


@pytest.mark.parametrize("is_time_varying", [False, True])
def test_tiny_kernel_fallback_warns(
    magmap_instance, observation_time_array, cosmology, is_time_varying
):
    """Tests the line "res_k = np.array([[1.0]])" in both the static and time-
    varying branches of MicrolensingLightCurve.

    If the source morphology kernel is smaller than one magnification
    map pixel, the code should trigger the fallback and emit a
    UserWarning.
    """

    # dummy morphology class that always returns a tiny kernel, to trigger the fallback
    class TinyMorphology:
        def __init__(self, time_varying):
            self.is_time_varying = time_varying

        @property
        def kernel_map(self):
            return np.ones((5, 5)) / 25.0

        @property
        def pixel_scale_m(self):
            return 1e-10

        def get_time_dependent_kernel_maps(self, times):
            return [np.ones((5, 5)) / 25.0 for _ in times], [1e-10] * len(times)

    ml_lc = MicrolensingLightCurve(
        magnification_map=magmap_instance,
        observation_time_array=observation_time_array,
        point_source_morphology="gaussian",
        kwargs_source_morphology={},
        source_morphology_instance=TinyMorphology(is_time_varying),
    )

    with pytest.warns(UserWarning, match="treating as a point source"):
        lcs, _, _ = ml_lc.generate_lightcurves(
            source_redshift=0.5, cosmo=cosmology, lightcurve_type="magnification"
        )

    assert isinstance(lcs[0], np.ndarray)
    assert not np.any(np.isnan(lcs[0]))
