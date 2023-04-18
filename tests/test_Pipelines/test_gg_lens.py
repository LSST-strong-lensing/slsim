import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from sim_pipeline.gg_lens import GGLens

class TestGGLens(object):
    @pytest.fixture
    def setup_method(self):
        blue_one = Table.read('data/Skypy/blue_one.fits', format='fits')
        red_one = Table.read('data/Skypy/red_one.fits', format='fits')
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = blue_one
        deflector_dict = red_one
        self.gg_lens = GGLens(source_dict, deflector_dict, cosmo)

    '''  
    def test_einstein_radius(self):
        einstein_radius = self.gg_lens.einstein_radius()
        assert einstein_radius == ????
  
    def test_deflector_ellipticity(self):
        e1_light, e2_light, e1_mass, e2_mass = self.gg_lens.deflector_ellipticity()
        assert e1_light == ?????
        assert e2_light == ?????
        assert e1_mass == ?????
        assert e2_mass == ?????
    
    def test_los_linear_distortions(self):
        gamma1, gamma2, kappa = self.gg_lens.los_linear_distortions()
        assert gamma1 == ?????
        assert gamma2 == ?????
        assert kappa == ?????
  '''
    def test_deflector_magnitude(self):
        band = 'g'
        deflector_magnitude = self.gg_lens.deflector_magnitude(band)
        assert isinstance(deflector_magnitude, float)
        assert pytest.approx(deflector_magnitude,rel=1e-3) == 30.780194

    def test_source_magnitude(self):
        band = 'g'
        source_magnitude = self.gg_lens.source_magnitude(band)
        assert pytest.approx(source_magnitude, rel=1e-3) == 26.4515655

    def test_lenstronomy_kwargs(self):
        band = 'g'
        kwargs_model, kwargs_params = self.gg_lens.lenstronomy_kwargs(band)
        assert 'source_light_model_list' in kwargs_model
        assert 'lens_light_model_list' in kwargs_model
        assert 'lens_model_list' in kwargs_model
        assert 'kwargs_lens' in kwargs_params
        assert 'kwargs_source' in kwargs_params
        assert 'kwargs_lens_light' in kwargs_params
        assert len(kwargs_params['kwargs_lens']) == 3
        assert len(kwargs_params['kwargs_source']) == 1
        assert len(kwargs_params['kwargs_lens_light']) == 1

if __name__ == '__main__':
    pytest.main()

