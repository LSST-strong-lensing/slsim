from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.galaxy_cluster_lenses import GalaxyClusterLenses
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
import pytest


def galaxy_list():
    sky_area = Quantity(value=0.05, unit="deg2")
    pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
    return pipeline.red_galaxies, pipeline.blue_galaxies


@pytest.fixture
def galaxy_cluster_lenses():
    red_galaxies = galaxy_list()[0]
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    return GalaxyClusterLenses(
        red_galaxies,  # TODO: replace by haloes and subhaloes
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


def test_deflector_number_draw_deflector(galaxy_cluster_lenses):
    # TODO: test after implementing draw_deflector
    galaxy_pop = galaxy_cluster_lenses
    num_deflectors = galaxy_pop.deflector_number()
    galaxy_pop.draw_deflector()
    # deflector = galaxy_pop.draw_deflector()
    # assert deflector["z"] != 0
    assert num_deflectors >= 0


if __name__ == "__main__":
    pytest.main()
