from slsim.Pipelines.halos_pipeline import HalosSkyPyPipeline
from astropy.cosmology import default_cosmology


class TestHalosSkyPyPipeline(object):
    def setup_method(self):
        sky_area = 0.001
        self.pipeline = HalosSkyPyPipeline(
            skypy_config=None, sky_area=sky_area, m_min=1.0e11, m_max=1.0e15, z_max=4.00
        )

    def test_halos(self):
        halos = self.pipeline.halos
        assert halos[0]["z"] > 0
        assert halos[0]["mass"] != halos[1]["mass"]


def test_cosmology_initialization():
    sky_area = 0.001
    galaxy_cosmo0 = default_cosmology.get()
    pipeline0 = HalosSkyPyPipeline(cosmo=galaxy_cosmo0, sky_area=sky_area)
    halos0 = pipeline0.halos
    assert halos0[0]["z"] > 0
