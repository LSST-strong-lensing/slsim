from astropy.cosmology import FlatLambdaCDM
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from slsim.Sources.galaxies import Galaxies
from slsim.Sources.galaxies import galaxy_projected_eccentricity
import pytest


class TestGalaxies(object):
    def setup_method(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
        self.galaxy_list = pipeline.red_galaxies
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.galaxies = Galaxies(
            galaxy_list=self.galaxy_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
        )

    def test_source_number(self):
        number = self.galaxies.source_number()
        assert number > 0

    def test_draw_source(self):
        galaxy = self.galaxies.draw_source()
        assert len(galaxy) > 0


def test_galaxy_projected_eccentricity():
    e1, e2 = galaxy_projected_eccentricity(0)
    assert e1 == 0
    assert e2 == 0


if __name__ == "__main__":
    pytest.main()
