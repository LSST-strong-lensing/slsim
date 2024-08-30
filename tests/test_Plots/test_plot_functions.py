import pytest

import numpy as np
import matplotlib.pyplot as plt
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors

from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.lens_pop import LensPop
from slsim.Plots.plot_functions import (
    create_image_montage_from_image_list,
    plot_montage_of_random_injected_lens,
)
from slsim.image_simulation import sharp_image


@pytest.fixture
def quasar_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.01, unit="deg2")

    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=None,
        sky_area=sky_area,
        filters=None,
    )

    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut={},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    quasar_galaxies = sources.QuasarCatalog.quasar_galaxies_simple(**{})
    source_galaxies = sources.PointPlusExtendedSources(
        point_plus_extended_sources_list=quasar_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut={},
        variability_model="sinusoidal",
        kwargs_variability_model={"amp", "freq"},
    )

    pes_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
    )

    return pes_lens_pop


def test_create_image_montage_from_image_list(quasar_lens_pop_instance):
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    image_list = []
    for i in range(6):
        lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
        image_list.append(
            sharp_image(
                lens_class=lens_class,
                band="i",
                mag_zero_point=27,
                delta_pix=0.2,
                num_pix=101,
            )
        )
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
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t, band=band1)
    fig4 = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t, band=band2)
    fig5 = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t, band=band3)

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
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    image_list = []
    for i in range(6):
        lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
        image_list.append(
            sharp_image(
                lens_class=lens_class,
                band="i",
                mag_zero_point=27,
                delta_pix=0.2,
                num_pix=101,
            )
        )

    num_rows = 2
    num_cols = 2
    fig = plot_montage_of_random_injected_lens(
        image_list=image_list, num=4, n_horizont=num_rows, n_vertical=num_cols
    )
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]


if __name__ == "__main__":
    pytest.main()
