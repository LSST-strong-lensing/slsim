import pytest
import pkg_resources
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import sim_pipeline
from sim_pipeline.gg_lens import GGLens

import os
class TestGGLens(object):
    @pytest.fixture(scope='class')
    def gg_lens(self):
        path = os.path.dirname(sim_pipeline.__file__)
        module_path, _ = os.path.split(path)
        blue_one = Table.read(os.path.join(module_path, 'data/Skypy/blue_one_modified.fits'), format='fits')
        red_one = Table.read(os.path.join(module_path, 'data/Skypy/red_one_modified.fits'), format='fits')
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        source_dict = blue_one
        deflector_dict = red_one
        return GGLens(source_dict=source_dict, deflector_dict=deflector_dict, cosmo=cosmo)

    def test_deflector_ellipticity(self, gg_lens):
        e1_light, e2_light, e1_mass, e2_mass = gg_lens.deflector_ellipticity()
        assert e1_light == -0.05661955320450283
        assert e2_light == 0.08738390223219591
        assert e1_mass == -0.08434700688970058
        assert e2_mass == 0.09710653297997263

    def test_deflector_magnitude(self, gg_lens):
        band = 'g'
        deflector_magnitude = gg_lens.deflector_magnitude(band)
        assert isinstance(deflector_magnitude[0], float)
        assert pytest.approx(deflector_magnitude[0],rel=1e-3) == 26.4515655

    def test_source_magnitude(self, gg_lens):
        band = 'g'
        source_magnitude = gg_lens.source_magnitude(band)
        assert pytest.approx(source_magnitude[0], rel=1e-3) == 30.780194

if __name__ == '__main__':
    pytest.main()

