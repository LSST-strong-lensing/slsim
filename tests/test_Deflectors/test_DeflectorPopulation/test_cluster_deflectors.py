from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology as colossus_cosmo
from slsim.Deflectors.DeflectorPopulation.cluster_deflectors import ClusterDeflectors
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Table, vstack
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
    module_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
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
    cluster_pop = cluster_deflectors_instance
    num_deflectors = cluster_pop.deflector_number()
    assert num_deflectors >= 0


def test_draw_deflector(cluster_deflectors_instance):
    cluster_pop = cluster_deflectors_instance
    deflector = cluster_pop.draw_deflector()
    cluster = cluster_pop.draw_cluster(index=0)
    members = cluster_pop.draw_members(cluster_id=cluster["cluster_id"])
    # test if the properties of the deflector are
    # as expected from the input catalog
    assert (deflector.redshift > 0.2) and (deflector.redshift < 1.0)
    assert (deflector.halo_properties[0] > 1e12) and (
        deflector.halo_properties[0] < 3e15
    )
    assert (deflector.halo_properties[1] > 1) and (deflector.halo_properties[1] < 15)
    assert (len(members) >= 1) and (len(members) < 100)


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


def test_with_centers(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    members_catalog["center_x"] = 0.0
    members_catalog["center_y"] = 0.0
    members_catalog.remove_column("ra")
    members_catalog.remove_column("dec")
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    cluster_pop = ClusterDeflectors(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    cluster = cluster_pop.draw_cluster(index=0)
    members = cluster_pop.draw_members(cluster_id=cluster["cluster_id"])
    assert members["center_x"][0] == 0.0


def test_missing_magnitudes(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    for col in members_catalog.colnames:
        if "mag" in col:
            members_catalog.remove_column(col)
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


def test_long_galaxy_list(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    red_galaxies = vstack([red_galaxies for _ in range(30)])
    assert len(red_galaxies) > 10000
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    cluster_pop = ClusterDeflectors(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    num_deflectors = cluster_pop.deflector_number()
    assert num_deflectors >= 0


def test_cosmo_Ob0(cluster_deflectors_input):
    cluster_catalog, members_catalog, red_galaxies = cluster_deflectors_input
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo_Ob0_zero = FlatLambdaCDM(H0=70, Om0=0.3)  # Ob0 defaults to 0
    cosmo_Ob0_none = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None)
    cosmo_Ob0_nonzero = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sky_area = Quantity(value=0.005, unit="deg2")
    ClusterDeflectors(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo_Ob0_zero,
        sky_area=sky_area,
    )
    assert colossus_cosmo.current_cosmo.Ob0 == 0.04897
    ClusterDeflectors(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo_Ob0_none,
        sky_area=sky_area,
    )
    assert colossus_cosmo.current_cosmo.Ob0 == 0.04897
    ClusterDeflectors(
        cluster_catalog,
        members_catalog,
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo_Ob0_nonzero,
        sky_area=sky_area,
    )
    assert colossus_cosmo.current_cosmo.Ob0 == 0.05


if __name__ == "__main__":
    pytest.main()
