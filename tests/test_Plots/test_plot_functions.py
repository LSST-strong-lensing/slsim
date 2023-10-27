import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from slsim.lens_pop import LensPop
from astropy.units import Quantity
from slsim.Plots.plot_functions import (
    create_image_montage_from_image_list,
    plot_montage_of_random_injected_lens,
)
from slsim.image_simulation import sharp_image
import pytest


@pytest.fixture
def quasar_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    return LensPop(
        deflector_type="all-galaxies",
        source_type="quasars",
        kwargs_deflector_cut=None,
        kwargs_source_cut=None,
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
    )


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
    t = np.linspace(0, 10, 6)
    fig = create_image_montage_from_image_list(
        num_rows=num_rows, num_cols=num_cols, images=image_list, time=t
    )
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]


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
