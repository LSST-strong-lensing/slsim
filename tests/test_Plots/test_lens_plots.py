import pytest
from slsim.lens_pop import LensPop
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Plots.lens_plots import LensingPlots
import matplotlib.pyplot as plt


def gg_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    return LensPop(sky_area=sky_area, cosmo=cosmo)


@pytest.fixture
def lensing_plots():
    lens_pop = gg_lens_pop_instance()
    return LensingPlots(lens_pop, num_pix=64, coadd_years=10)


def test_rgb_image(lensing_plots):
    lens_pop = gg_lens_pop_instance()
    kwargs_lens_cut = {}
    lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    image_rgb = lensing_plots.rgb_image(lens_class, rgb_band_list, add_noise=add_noise)

    assert isinstance(image_rgb, np.ndarray)
    assert image_rgb.shape == (
        lensing_plots.num_pix,
        lensing_plots.num_pix,
        3,
    )


def test_plot_montage(lensing_plots):
    rgb_band_list = ["r", "g", "i"]
    add_noise = True
    n_horizont = 2
    n_vertical = 2
    kwargs_lens_cut_plot = None
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
