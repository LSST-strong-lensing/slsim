from sim_pipeline.Pipelines.halos_pipeline import HalosSkyPyPipeline


class TestHalosSkyPyPipeline(object):
    def setup_method(self):
        from astropy.units import Quantity

        sky_area = 0.00005
        self.pipeline = HalosSkyPyPipeline(
            skypy_config=None, sky_area=sky_area, m_min=1.0e13, m_max=1.0e15, z_max=4.00
        )

    def test_halos(self):
        halos = self.pipeline.halos
        assert halos[0]["z"] > 0
        assert halos[0]["mass"] != halos[1]["mass"]
