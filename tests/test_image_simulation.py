import os
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from sim_pipeline.gg_lens import GGLens
from sim_pipeline.image_simulation import simulate_image, sharp_image


class TestImageSimulation(object):

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
            gg_lens = GGLens(source_dict=self.source_dict, deflector_dict=self.deflector_dict, cosmo=cosmo)
            if gg_lens.validity_test():
                self.gg_lens = gg_lens
                break

    def test_simulate_image(self):
        image = simulate_image(lens_class=self.gg_lens, band='g', num_pix=100, add_noise=True, observatory='LSST')
        assert len(image) == 100

    def test_sharp_image(self):
        image = sharp_image(lens_class=self.gg_lens, band='g', mag_zero_point=30, delta_pix=0.05, num_pix=100,
                            with_deflector=True)
        assert len(image) == 100
