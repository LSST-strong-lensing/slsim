import pytest
from sim_pipeline.galaxy_galaxy_lens_pop import GalaxyGalaxyLensPop, draw_test_area
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from sim_pipeline.Plots.galaxy_galaxy_plots import GalaxyGalaxyLensingPlots


class test_GalaxyGalaxyLensingPlots(object):

    @pytest.fixture(scope="class")
    def gg_lensing_instance(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.1, unit='deg2')
        self.gg_lens_pop = GalaxyGalaxyLensPop(sky_area=sky_area, cosmo=cosmo)
        self.gg_plot = GalaxyGalaxyLensingPlots(self.gg_lens_pop, num_pix=64, coadd_years=10)
    def test_rgb_image(self):
        image = self.gg_plot.rgb_image(lens_class = self.gg_lens_pop, rgb_band_list = ['r', 'g', 'i'], add_noise=True)
        assert len(image) == 64
    
    def test_plot_montage(self):
         kwargs_lens_cut_plot = {'min_image_separation': 0.8, 'max_image_separation': 10, 
                        'mag_arc_limit': {'g': 23, 'r': 23, 'i': 23}}
         fig, axes = self.gg_plot.plot_montage(rgb_band_list=['i', 'r', 'g'], add_noise=True, n_horizont=1, 
                                 n_vertical=1, kwargs_lens_cut=kwargs_lens_cut_plot)
         assert len(axes) == 1

if __name__ == "__main__":
    pytest.main()
         

    