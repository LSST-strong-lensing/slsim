from sim_pipeline.Pipelines.halos_pipeline import HalosSkyPyPipeline

class TestHalosSkyPyPipeline(object):

    def setup_method(self):
        from astropy.units import Quantity
        sky_area = Quantity(value=0.00001, unit='deg2')
        self.pipeline = HalosSkyPyPipeline(skypy_config=None, sky_area=sky_area)

    def test_halos(self):
        halos = self.pipeline.halos
        assert halos[0]['z'] > 0
        assert halos[0]['mass'] != halos[1]['mass']
