import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from sim_pipeline.galaxy_galaxy_lens_pop import GalaxyGalaxyLensPop
from sim_pipeline.Sources.source_variability.variability import sinusoidal_variability
from astropy.units import Quantity
from sim_pipeline.Plots.plot_functions import plot_montage_of_random_injected_lens
from sim_pipeline.image_simulation import sharp_image
import pytest

@pytest.fixture
def quasar_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    return GalaxyGalaxyLensPop(deflector_type="all-galaxies",
    source_type="quasars",
    kwargs_deflector_cut=None,
    kwargs_source_cut=None,
    kwargs_mass2light=None,
    skypy_config=None,
    sky_area=sky_area,
    cosmo=cosmo)



def test_plot_montage_of_random_injected_lens(quasar_lens_pop_instance):
    kwargs_lens_cut={'min_image_separation': 0.8, 'max_image_separation': 10}
    image_list = []
    for i in range(4):
        lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
        image_list.append(sharp_image(lens_class=lens_class, band='i', 
                                mag_zero_point=27, delta_pix=0.2, num_pix=101))

    plots = plot_montage_of_random_injected_lens(
        image_list = image_list, num = 4,
        n_horizont=2,
        n_vertical=2)
    assert plots is None

if __name__ == "__main__":
    pytest.main()