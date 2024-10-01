import pytest

import numpy as np
import matplotlib.pyplot as plt
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors

from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.lens import Lens
from slsim.Plots.plot_functions import (
    create_image_montage_from_image_list,
    plot_montage_of_random_injected_lens,
)
from slsim.image_simulation import sharp_image
from astropy.table import Table
import os


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
        pes_lens = Lens(
            source_dict=source_dict,
            deflector_dict=deflector_dict,
            source_type="point_plus_extended",
            variability_model="sinusoidal",
            kwargs_variability={"amp", "freq"},
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
    assert isinstance(fig, plt.Figure)
    assert fig.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]
    assert isinstance(fig2, plt.Figure)
    assert fig2.get_size_inches()[0] == np.array([num_cols * 3, num_rows * 3])[0]


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


if __name__ == "__main__":
    pytest.main()
