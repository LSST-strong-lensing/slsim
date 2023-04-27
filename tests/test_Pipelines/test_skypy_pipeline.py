from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline


class TestSkyPyPipeline(object):

    def setup_method(self):
        self.pipeline = SkyPyPipeline(skypy_config=None, f_sky=0.1)

    def test_blue_galaxies(self):
        blue_galaxies = self.pipeline.blue_galaxies
        assert blue_galaxies[0]['z'] > 0

    def test_red_galaxies(self):
        red_galaxies = self.pipeline.red_galaxies
        assert red_galaxies[0]['z'] > 0
