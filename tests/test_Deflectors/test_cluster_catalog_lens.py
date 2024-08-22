from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.cluster_catalog_lens import ClusterCatalogLens
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Table
import pytest
import os


def galaxy_list():
    sky_area = Quantity(value=0.005, unit="deg2")
    pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
    red_gal = pipeline.red_galaxies
    return red_gal


@pytest.fixture
def cluster_catalog_lens():
    red_galaxies = galaxy_list()

    path = os.path.dirname(__file__)
    module_path = os.path.dirname(os.path.dirname(path))
    cluster_catalog = Table.read(
        os.path.join(module_path, "data/redMaPPer/clusters_example.fits")
    )
    members_catalog = Table.read(
        os.path.join(module_path, "data/redMaPPer/members_example.fits")
    )

    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    return ClusterCatalogLens(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


def test_deflector_number(cluster_catalog_lens):
    galaxy_pop = cluster_catalog_lens
    num_deflectors = galaxy_pop.deflector_number()
    assert num_deflectors >= 0


def test_draw_deflector(cluster_catalog_lens):
    galaxy_pop = cluster_catalog_lens
    deflector = galaxy_pop.draw_deflector()
    assert deflector["z"] != 0
    assert deflector["halo_mass"] > 0
    assert deflector["concentration"] > 0
    assert len(deflector["subhalos"]) > 0


if __name__ == "__main__":
    pytest.main()
