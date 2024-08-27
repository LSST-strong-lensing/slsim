from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.cluster_deflectors import ClusterDeflectors
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
def cluster_deflectors_input():
    red_galaxies = galaxy_list()

    path = os.path.dirname(__file__)
    module_path = os.path.dirname(os.path.dirname(path))
    cluster_catalog = Table.read(
        os.path.join(module_path, "data/redMaPPer/clusters_example.fits")
    )
    members_catalog = Table.read(
        os.path.join(module_path, "data/redMaPPer/members_example.fits")
    )

    return cluster_catalog, members_catalog, red_galaxies


@pytest.fixture
def cluster_deflectors_instance(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    return ClusterDeflectors(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


def test_deflector_number(cluster_deflectors_instance):
    galaxy_pop = cluster_deflectors_instance
    num_deflectors = galaxy_pop.deflector_number()
    assert num_deflectors >= 0


def test_draw_deflector(cluster_deflectors_instance):
    galaxy_pop = cluster_deflectors_instance
    deflector = galaxy_pop.draw_deflector()
    # test if the properties of the deflector are
    # as expected from the input catalog
    assert (deflector["z"] > 0.2) and (deflector["z"] < 1.0)
    assert (deflector["halo_mass"] > 1e12) and (deflector["halo_mass"] < 3e15)
    assert (deflector["concentration"] > 1) and (deflector["concentration"] < 15)
    assert (len(deflector["subhalos"]) >= 1) and (len(deflector["subhalos"]) < 100)


def test_missing_id(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    cluster_catalog.remove_column("cluster_id")
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    with pytest.raises(ValueError):
        ClusterDeflectors(
            cluster_catalog,
            members_catalog,
            red_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
        )


def test_missing_richness(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    cluster_catalog.remove_column("richness")
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    with pytest.raises(ValueError):
        ClusterDeflectors(
            cluster_catalog,
            members_catalog,
            red_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
        )


def test_missing_redshift(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    cluster_catalog.remove_column("z")
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    with pytest.raises(ValueError):
        ClusterDeflectors(
            cluster_catalog,
            members_catalog,
            red_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
        )


def test_missing_ra_dec(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    members_catalog.remove_column("ra")
    members_catalog.remove_column("dec")
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    with pytest.raises(ValueError):
        ClusterDeflectors(
            cluster_catalog,
            members_catalog,
            red_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
        )


if __name__ == "__main__":
    pytest.main()
