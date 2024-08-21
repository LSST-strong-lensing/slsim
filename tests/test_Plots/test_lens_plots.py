import pytest

import numpy as np
import matplotlib.pyplot as plt
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors

from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.lens_pop import LensPop
from slsim.Plots.lens_plots import LensingPlots


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
    )

    lenspop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
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
    )
    assert isinstance(fig, plt.Figure)
    assert len(axes) == n_vertical
    assert len(axes[0]) == n_horizont


if __name__ == "__main__":
    pytest.main()
