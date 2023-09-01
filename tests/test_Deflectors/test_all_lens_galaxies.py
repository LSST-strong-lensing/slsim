from astropy.cosmology import FlatLambdaCDM
from sim_pipeline.Deflectors.all_lens_galaxies import AllLensGalaxies, fill_table, vel_disp_abundance_matching
from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
import pytest

class test_all_galaxies(object):
        def setup_method(self): 
            self.sky_area = Quantity(value=0.1, unit='deg2')
            pipeline = SkyPyPipeline(skypy_config=None, sky_area=self.sky_area, filters=None)
            kwargs_deflector_cut = {}
            kwargs_mass2light = {}
            self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            self.lens_galaxies = AllLensGalaxies(pipeline.red_galaxies, pipeline.blue_galaxies,
                                                    kwargs_cut=kwargs_deflector_cut, kwargs_mass2light=kwargs_mass2light,
                                                    cosmo = self.cosmo, sky_area = self.fsky_area)
            
        def test_deflector_number(self):
            number = self.lens_galaxies.deflector_number()
            assert number > 0

        
        def test_draw_deflector(self):
            deflector_drawn = self.lens_galaxies.draw_deflector()
            assert deflector_drawn > 0  

def test_fill_table():
        pipeline = SkyPyPipeline(skypy_config=None, sky_area=Quantity(value=0.1, unit='deg2'), filters=None)
        list = fill_table(pipeline.red_galaxies)
        assert len(list) > 0

def test_vel_disp_abundance_matching():
        pipeline = SkyPyPipeline(skypy_config=None, sky_area=Quantity(value=0.1, unit='deg2'), filters=None)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        vdp = vel_disp_abundance_matching(pipeline.red_galaxies, z_max=0.5, sky_area=Quantity(value=0.1, unit='deg2'), cosmo=cosmo)
        assert vdp(10**10) != 0

if __name__ == '__main__':
    pytest.main()
