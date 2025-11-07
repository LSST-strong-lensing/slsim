import pytest

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from slsim.Lenses.lens import Lens
from slsim.Plots.plot_functions import (
    create_image_montage_from_image_list,
    plot_montage_of_random_injected_lens,
    create_montage,
    plot_lightcurves,
)
from slsim.ImageSimulation.image_simulation import sharp_image
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from astropy.table import Table
import os
from slsim.Plots.plot_functions import plot_lightcurves_and_magmap
from slsim.Microlensing.lightcurve import MicrolensingLightCurve
from slsim.Microlensing.magmap import MagnificationMap
from slsim.Plots.plot_functions import plot_magnification_map


@pytest.fixture
def quasar_lens_pop_instance():
    path = os.path.dirname(__file__)
    new_path = os.path.dirname(path)
    source_dict = Table.read(
        os.path.join(new_path, "TestData/source_dict_ps.fits"), format="fits"
    )
    deflector_dict = Table.read(
        os.path.join(new_path, "TestData/deflector_dict_ps.fits"), format="fits"
    )

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    while True:
        variable_agn_kwarg_dict = {
            "length_of_light_curve": 500,
            "time_resolution": 1,
            "log_breakpoint_frequency": 1 / 20,
            "low_frequency_slope": 1,
            "high_frequency_slope": 3,
            "standard_deviation": 0.9,
        }
        kwargs_quasar = {
            "variability_model": "light_curve",
            "kwargs_variability": {"agn_lightcurve", "i", "r"},
            "agn_driving_variability_model": "bending_power_law",
            "agn_driving_kwargs_variability": variable_agn_kwarg_dict,
            "lightcurve_time": np.linspace(0, 1000, 1000),
        }
        source = Source(
            source_dict=source_dict,
            cosmo=cosmo,
            point_source_type="quasar",
            extended_source_type="single_sersic",
            **source_dict,
            **kwargs_quasar,
        )
        deflector = Deflector(
            deflector_type="EPL_SERSIC",
            **deflector_dict,
        )
        pes_lens = Lens(
            source_class=source,
            deflector_class=deflector,
            cosmo=cosmo,
        )
        if pes_lens.validity_test():
            pes_lens = pes_lens
            break
    return pes_lens


def test_create_image_montage_from_image_list(quasar_lens_pop_instance):
    lens_class = quasar_lens_pop_instance
    image = sharp_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
    )
    image_list = [image, image, image, image, image, image]

    num_rows = 2
    num_cols = 3

    # Create different types of input for "band" to test the response of the function
    band1 = "i"
    band2 = ["i"] * len(image_list)
    band3 = None

    t = np.linspace(0, 10, 6)
    fig = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t
    )
    fig2 = create_image_montage_from_image_list(
        num_rows=num_rows,
        num_cols=num_cols,
        images=image_list,
        time=t,
        image_type="dp0",
    )
    fig3 = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t, band=band1
    )
    fig4 = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t, band=band2
    )
    fig5 = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t, band=band3
    )

    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]
    assert isinstance(fig2, plt.Figure)
    assert fig2.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]
    assert isinstance(fig3, plt.Figure)
    assert fig3.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]
    assert isinstance(fig4, plt.Figure)
    assert fig4.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]
    assert isinstance(fig5, plt.Figure)
    assert fig5.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]


def test_plot_montage_of_random_injected_lens(quasar_lens_pop_instance):
    lens_class = quasar_lens_pop_instance
    image = sharp_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
    )
    image_list = [image, image, image, image, image, image]

    num_rows = 2
    num_cols = 2
    fig = plot_montage_of_random_injected_lens(
        image_list=image_list, num=4, n_horizont=num_rows, n_vertical=num_cols
    )
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]


def test_create_montage_basics():
    images = [
        np.random.rand(5, 5),
        np.random.rand(5, 5),
        np.random.rand(5, 5),
        np.random.rand(5, 5),
    ]
    montage = create_montage(images)

    # Check shape
    assert montage.shape == (5, 15)  # 1 row, 3 images wide

    # Check normalization range
    assert np.min(montage) >= 0
    assert np.max(montage) <= 1


def test_create_montage_specified_grid():
    images = [
        np.random.rand(5, 5),
        np.random.rand(5, 5),
        np.random.rand(5, 5),
    ]
    grid_size = (1, 3)
    montage = create_montage(images, grid_size=grid_size)

    # Check shape
    assert montage.shape == (5, 15)  # 1 row, 3 images wide


def test_plot_lightcurves():
    data = {
        "magnitudes": {
            "mag_image_1": {"g": np.random.rand(5), "r": np.random.rand(5)},
            "mag_image_2": {"g": np.random.rand(5), "r": np.random.rand(5)},
        },
        "errors_low": {
            "mag_error_image_1_low": {"g": np.random.rand(5), "r": np.random.rand(5)},
            "mag_error_image_2_low": {"g": np.random.rand(5), "r": np.random.rand(5)},
        },
        "errors_high": {
            "mag_error_image_1_high": {"g": np.random.rand(5), "r": np.random.rand(5)},
            "mag_error_image_2_high": {"g": np.random.rand(5), "r": np.random.rand(5)},
        },
        "obs_time": {"g": np.arange(5), "r": np.arange(5)},
        "image_lists": {
            "g": [np.random.rand(10, 10) for _ in range(3)],
            "r": [np.random.rand(10, 10) for _ in range(3)],
        },
    }
    data2 = {
        "magnitudes": {
            "mag_image_1": {"g": np.random.rand(5)},
            "mag_image_2": {"g": np.random.rand(5)},
        },
        "errors_low": {
            "mag_error_image_1_low": {"g": np.random.rand(5)},
            "mag_error_image_2_low": {"g": np.random.rand(5)},
        },
        "errors_high": {
            "mag_error_image_1_high": {"g": np.random.rand(5)},
            "mag_error_image_2_high": {"g": np.random.rand(5)},
        },
        "obs_time": {"g": np.arange(5)},
        "image_lists": {"g": [np.random.rand(10, 10) for _ in range(3)]},
    }

    fig = plot_lightcurves(data)
    fig3 = plot_lightcurves(data2)
    ax3 = fig3.get_axes()
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert len(ax3) == 2


#### MICROLENSING TESTS ####

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


# --- Test Plotting (Execution Check Only) ---


# Test plotting (execution only)
@pytest.mark.parametrize("plot_magnitude", [True, False])
def test_plot_magnification_map_runs(magmap_instance, plot_magnitude):
    """Tests that the plotting function runs without error."""
    fig, ax = plt.subplots()
    try:
        plot_magnification_map(magmap_instance, ax=ax, plot_magnitude=plot_magnitude)
        assert len(ax.images) > 0
        assert ax.get_xlabel() == "$x / \\theta_★$"
        assert ax.get_ylabel() == "$y / \\theta_★$"
        assert len(fig.axes) > 1  # Check colorbar axes was added
    except Exception as e:
        pytest.fail(f"plot_magnification_map raised an exception: {e}")
    finally:
        plt.close(fig)


def test_plot_magnification_map_runs_no_ax(magmap_instance):
    """Tests plotting function runs without error when ax is None."""
    try:
        plot_magnification_map(magmap_instance, ax=None, plot_magnitude=True)
        assert plt.gcf().number > 0  # Check a figure was created
    except Exception as e:
        pytest.fail(f"plot_magnification_map raised an exception: {e}")
    finally:
        plt.close("all")


def test_plot_lightcurves_and_magmap_runs_magnitude(ml_lc_gaussian, cosmology):
    """Tests plotting function runs without error (magnitude)."""
    num_lc = 2
    lcs, tracks, _time_arrays = ml_lc_gaussian.generate_lightcurves(
        0.5, cosmology, num_lightcurves=num_lc
    )
    ml_lc_gaussian.get_convolved_map()
    try:
        ax_return = plot_lightcurves_and_magmap(
            convolved_map=ml_lc_gaussian._convolved_map,
            lightcurves=lcs,
            time_duration_observer_frame=ml_lc_gaussian._time_duration_observer_frame,
            tracks=tracks,
            magmap_instance=ml_lc_gaussian._magnification_map,
            lightcurve_type="magnitude",
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
def test_plot_lightcurves_and_magmap_runs_magnification(ml_lc_agn_wave, cosmology):
    """Tests plotting function runs without error (magnification)."""
    num_lc = 1
    lcs, _tracks, _time_arrays = ml_lc_agn_wave.generate_lightcurves(
        0.5, cosmology, num_lightcurves=num_lc, lightcurve_type="magnification"
    )
    ml_lc_agn_wave.get_convolved_map()
    try:
        ax_return = plot_lightcurves_and_magmap(
            convolved_map=ml_lc_agn_wave._convolved_map,
            lightcurves=lcs,
            time_duration_observer_frame=ml_lc_agn_wave._time_duration_observer_frame,
            tracks=None,
            magmap_instance=ml_lc_agn_wave._magnification_map,
            lightcurve_type="magnification",
        )
        assert isinstance(ax_return, np.ndarray)
        assert all(isinstance(ax, plt.Axes) for ax in ax_return.flat)
    except Exception as e:
        pytest.fail(f"plot_lightcurves_and_magmap raised: {e}")
    finally:
        plt.close("all")


############################


if __name__ == "__main__":
    pytest.main()
