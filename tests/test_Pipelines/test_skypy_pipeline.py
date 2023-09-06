from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline


class TestSkyPyPipeline(object):
    def setup_method(self):
        from astropy.units import Quantity

        sky_area = Quantity(value=0.05, unit="deg2")
        self.pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area)

    def test_blue_galaxies(self):
        blue_galaxies = self.pipeline.blue_galaxies
        assert blue_galaxies[0]["z"] > 0

    def test_red_galaxies(self):
        red_galaxies = self.pipeline.red_galaxies
        assert red_galaxies[0]["z"] > 0
