import pytest
from sim_pipeline.galaxy_galaxy_lens_pop import GalaxyGalaxyLensPop
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from sim_pipeline.Plots.galaxy_galaxy_plots import GalaxyGalaxyLensingPlots
import matplotlib.pyplot as plt


def gg_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    return GalaxyGalaxyLensPop(sky_area=sky_area, cosmo=cosmo)


@pytest.fixture
def galaxy_galaxy_lensing_plots():
    lens_pop = gg_lens_pop_instance()
    return GalaxyGalaxyLensingPlots(lens_pop, num_pix=64, coadd_years=10)


def test_rgb_image(galaxy_galaxy_lensing_plots):
    lens_pop = gg_lens_pop_instance()
    kwargs_lens_cut = {}
    lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    image_rgb = galaxy_galaxy_lensing_plots.rgb_image(
        lens_class, rgb_band_list, add_noise=add_noise
    )

    assert isinstance(image_rgb, np.ndarray)
    assert image_rgb.shape == (
        galaxy_galaxy_lensing_plots.num_pix,
        galaxy_galaxy_lensing_plots.num_pix,
        3,
    )


def test_plot_montage(galaxy_galaxy_lensing_plots):
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    n_horizont = 2
    n_vertical = 2
    kwargs_lens_cut_plot = None
    fig, axes = galaxy_galaxy_lensing_plots.plot_montage(
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
