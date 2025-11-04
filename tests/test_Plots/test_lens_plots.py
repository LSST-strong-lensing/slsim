import pytest
import os

import numpy as np
import matplotlib.pyplot as plt
import slsim
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors

from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Lenses.lens_pop import LensPop
from slsim.Plots.lens_plots import LensingPlots
from slsim.Pipelines.roman_speclite import configure_roman_filters
from slsim.Pipelines.roman_speclite import filter_names
import speclite


@pytest.fixture
def gg_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")

    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=None,
        sky_area=sky_area,
        filters=None,
    )
    lens_galaxies = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut={},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut={},
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        extended_source_type="single_sersic",
    )

    lenspop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    return lenspop


def gg_roman_lens_pop_instance():

    # generate Roman filters
    configure_roman_filters()

    # import filter bands and make them recogniable in speclite
    roman_filters = filter_names()
    # filters are ['Roman-F062', 'Roman-F087', 'Roman-F106', 'Roman-F129', 'Roman-F158', 'Roman-F184', 'Roman-F146', 'Roman-F213']

    speclite.filters.load_filters(
        roman_filters[0],
        roman_filters[1],
        roman_filters[2],
        roman_filters[3],
        roman_filters[4],
        roman_filters[5],
        roman_filters[6],
        roman_filters[7],
    )

    path = os.path.dirname(slsim.__file__)
    module_path, _ = os.path.split(path)
    skypy_config = os.path.join(module_path, "data/SkyPy/roman-like.yml")

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")

    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=skypy_config,
        sky_area=sky_area,
        filters=None,
    )
    lens_galaxies = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut={},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut={},
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        source_size=None,
        extended_source_type="single_sersic",
    )

    lenspop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    return lenspop


def test_rgb_image(gg_lens_pop_instance):
    kwargs_lens_cut = {}
    lens_class = gg_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    lensing_plots = LensingPlots(gg_lens_pop_instance, num_pix=64, coadd_years=5)
    image_rgb = lensing_plots.rgb_image(lens_class, rgb_band_list, add_noise=add_noise)

    assert isinstance(image_rgb, np.ndarray)
    assert image_rgb.shape == (
        lensing_plots.num_pix,
        lensing_plots.num_pix,
        3,
    )


# NOTE: Galsim is required which is not supported on Windows
def test_roman_rgb_image():
    lens_pop = gg_roman_lens_pop_instance()
    kwargs_lens_cut = {}
    lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
    # Only F106 can be tested since external files are required for other bands
    rgb_band_list = ["F106", "F106", "F106"]
    lensing_plots = LensingPlots(lens_pop, num_pix=64, observatory="Roman")
    image_rgb = lensing_plots.rgb_image(lens_class, rgb_band_list)

    assert isinstance(image_rgb, np.ndarray)
    assert image_rgb.shape == (
        lensing_plots.num_pix,
        lensing_plots.num_pix,
        3,
    )


def test_plot_montage(gg_lens_pop_instance):
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    n_horizont = 2
    n_vertical = 2
    kwargs_lens_cut_plot = None
    lensing_plots = LensingPlots(gg_lens_pop_instance, num_pix=64, coadd_years=5)
    fig, axes = lensing_plots.plot_montage(
        rgb_band_list,
        add_noise=add_noise,
        n_horizont=n_horizont,
        n_vertical=n_vertical,
        kwargs_lens_cut=kwargs_lens_cut_plot,
        minimum=0,
        stretch=5,
        Q=5,
    )
    assert isinstance(fig, plt.Figure)
    assert len(axes) == n_vertical
    assert len(axes[0]) == n_horizont


def test_plot_montage_single_band(gg_lens_pop_instance):
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    n_horizont = 2
    n_vertical = 2
    kwargs_lens_cut_plot = {}
    n_total = n_horizont * n_vertical
    lensing_plots = LensingPlots(gg_lens_pop_instance, num_pix=64, coadd_years=5)
    gg_lens_list = [
        gg_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut_plot)
        for _ in range(n_total)
    ]
    fig, axes = lensing_plots.plot_montage(
        rgb_band_list,
        add_noise=add_noise,
        n_horizont=n_horizont,
        n_vertical=n_vertical,
        kwargs_lens_cut=kwargs_lens_cut_plot,
        single_band=True,
        lens_class_list=gg_lens_list,
    )
    assert isinstance(fig, plt.Figure)
    assert len(axes) == n_vertical
    assert len(axes[0]) == n_horizont


if __name__ == "__main__":
    pytest.main()
