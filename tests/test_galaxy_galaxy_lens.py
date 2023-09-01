import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from sim_pipeline.galaxy_galaxy_lens import (GalaxyGalaxyLens, image_separation_from_positions,
                                              theta_e_when_source_infinity)

import os


class TestGalaxyGalaxyLens(object):
    # pytest.fixture(scope='class')
    def setup_method(self):
        # path = os.path.dirname(sim_pipeline.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(os.path.join(path, 'TestData/blue_one_modified.fits'), format='fits')
        red_one = Table.read(os.path.join(path, 'TestData/red_one_modified.fits'), format='fits')
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        while True:
            gg_lens = GalaxyGalaxyLens(source_dict=self.source_dict, deflector_dict=self.deflector_dict, cosmo=cosmo)
            if gg_lens.validity_test():
                self.gg_lens = gg_lens
                break

    def test_deflector_ellipticity(self):
        e1_light, e2_light, e1_mass, e2_mass = self.gg_lens.deflector_ellipticity()
        assert pytest.approx(e1_light, rel=1e-3) == -0.05661955320450283
        assert pytest.approx(e2_light, rel=1e-3) == 0.08738390223219591
        assert pytest.approx(e1_mass, rel=1e-3) == -0.08434700688970058
        assert pytest.approx(e2_mass, rel=1e-3) == 0.09710653297997263

    def test_deflector_magnitude(self):
        band = 'g'
        deflector_magnitude = self.gg_lens.deflector_magnitude(band)
        assert isinstance(deflector_magnitude[0], float)
        assert pytest.approx(deflector_magnitude[0], rel=1e-3) == 26.4515655

    def test_source_magnitude(self):
        band = 'g'
        source_magnitude = self.gg_lens.extended_source_magnitude(band)
        assert pytest.approx(source_magnitude[0], rel=1e-3) == 30.780194

    def test_image_separation_from_positions(self):
        image_positions = self.gg_lens.image_positions()
        image_separation = image_separation_from_positions(image_positions)
        theta_E_infinity = theta_e_when_source_infinity(deflector_dict=self.deflector_dict)
        assert image_separation < 2 * theta_E_infinity

    def test_theta_e_when_source_infinity(self):
        theta_E_infinity = theta_e_when_source_infinity(deflector_dict=self.deflector_dict)
        # We expect that theta_E_infinity should be less than 15
        assert theta_E_infinity < 15

    def test_host_magnification(self):
        host_mag = self.gg_lens.host_magnification()
        assert host_mag > 0

    def test_deflector_stellar_mass(self):
        s_mass = self.gg_lens.deflector_stellar_mass()
        assert s_mass >= 10**5

    def test_deflector_velocity_dispersion(self):
        vdp = self.gg_lens.deflector_velocity_dispersion()
        assert vdp >= 10

    def test_los_linear_distortions(self):
        losd = self.gg_lens.los_linear_distortions()
        assert losd != None


if __name__ == '__main__':
    pytest.main()
