import copy

from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology as colossus_cosmo
from slsim.Deflectors.DeflectorPopulation.cluster_deflectors import ClusterDeflectors
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Table
import pytest
import os


class TestClusterDeflector:

    def setup_method(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        pipeline = SkyPyPipeline(
            skypy_config=None, sky_area=sky_area, filters=None, cosmo=cosmo
        )
        self.red_gal = pipeline.red_galaxies
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        self.galaxies = Galaxies(
            galaxy_list=self.red_gal,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut=kwargs_deflector_cut,
            catalog_type="skypy",
        )

        path = os.path.dirname(__file__)
        module_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        self.cluster_catalog = Table.read(
            os.path.join(module_path, "data/redMaPPer/clusters_example.fits")
        )
        self.members_catalog = Table.read(
            os.path.join(module_path, "data/redMaPPer/members_example.fits")
        )

        self.cluster_pop = ClusterDeflectors(
            self.cluster_catalog,
            self.members_catalog,
            galaxies=self.galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
        )

    def test_deflector_number(self):

        num_deflectors = self.cluster_pop.deflector_number()
        assert num_deflectors == 80

    def test_draw_deflector(self):

        deflector = self.cluster_pop.draw_deflector()
        cluster = self.cluster_pop.draw_cluster(index=0)
        members = self.cluster_pop._draw_members(cluster_id=cluster["cluster_id"])
        # test if the properties of the deflector are
        # as expected from the input catalog
        assert (deflector.redshift > 0.2) and (deflector.redshift < 1.0)
        assert (
            deflector.deflector(deflector_index=0).mass_properties["halo_mass"] > 1e12
        ) and (
            deflector.deflector(deflector_index=0).mass_properties["halo_mass"] < 3e15
        )
        assert (
            deflector.deflector(deflector_index=0).mass_properties["concentration"] > 1
        ) and (
            deflector.deflector(deflector_index=0).mass_properties["concentration"] < 15
        )
        assert (len(members) >= 1) and (len(members) < 100)

    def test_missing_id(self):
        cluster_catalog = copy.deepcopy(self.cluster_catalog)
        cluster_catalog.remove_column("cluster_id")
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        with pytest.raises(ValueError):
            ClusterDeflectors(
                cluster_catalog,
                self.members_catalog,
                self.galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

    def test_missing_richness(self):
        cluster_catalog = copy.deepcopy(self.cluster_catalog)
        cluster_catalog.remove_column("richness")

        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        with pytest.raises(ValueError):
            ClusterDeflectors(
                cluster_catalog,
                self.members_catalog,
                self.galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

    def test_missing_redshift(self):
        cluster_catalog = copy.deepcopy(self.cluster_catalog)
        cluster_catalog.remove_column("z")
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        with pytest.raises(ValueError):
            ClusterDeflectors(
                cluster_catalog,
                self.members_catalog,
                galaxies=self.galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

    def test_missing_ra_dec(self):
        members_catalog = copy.deepcopy(self.members_catalog)
        members_catalog.remove_column("ra")
        members_catalog.remove_column("dec")
        members_catalog.remove_column("center_x")
        members_catalog.remove_column("center_y")
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        with pytest.raises(ValueError):
            clusters = ClusterDeflectors(
                self.cluster_catalog,
                members_catalog,
                galaxies=self.galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )
            clusters.draw_deflector()

    def test_with_centers(self):
        members_catalog = copy.deepcopy(self.members_catalog)
        members_catalog["center_x"] = 0.0
        members_catalog["center_y"] = 0.0
        members_catalog.remove_column("ra")
        members_catalog.remove_column("dec")
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        cluster_pop = ClusterDeflectors(
            self.cluster_catalog,
            members_catalog,
            galaxies=self.galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        cluster = cluster_pop.draw_cluster(index=0)
        members = cluster_pop._draw_members(cluster_id=cluster["cluster_id"])
        assert members["center_x"][0] == 0.0

    def test_missing_magnitudes(self):
        members_catalog = copy.deepcopy(self.members_catalog)
        for col in members_catalog.colnames:
            if "mag" in col:
                members_catalog.remove_column(col)
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        sky_area = Quantity(value=0.005, unit="deg2")
        with pytest.raises(ValueError):
            ClusterDeflectors(
                self.cluster_catalog,
                members_catalog,
                self.galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

    def test_cosmo_Ob0(self):
        kwargs_deflector_cut = {}
        kwargs_mass2light = {}
        cosmo_Ob0_zero = FlatLambdaCDM(H0=70, Om0=0.3)  # Ob0 defaults to 0
        cosmo_Ob0_nonzero = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        sky_area = Quantity(value=0.005, unit="deg2")
        ClusterDeflectors(
            self.cluster_catalog,
            self.members_catalog,
            self.galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo_Ob0_zero,
            sky_area=sky_area,
        )
        assert colossus_cosmo.current_cosmo.Ob0 == 0.04897

        ClusterDeflectors(
            self.cluster_catalog,
            self.members_catalog,
            self.galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo_Ob0_nonzero,
            sky_area=sky_area,
        )
        assert colossus_cosmo.current_cosmo.Ob0 == 0.05

    def test_get_deflector(self):
        cluster_pop = self.cluster_pop
        cluster = cluster_pop.draw_cluster(index=0)
        deflector = cluster_pop.draw_deflector(deflector_index=0)
        members = cluster_pop._draw_members(cluster_id=cluster["cluster_id"])
        # test if the properties of the deflector are
        # as expected from the input catalog
        assert (deflector.redshift > 0.2) and (deflector.redshift < 1.0)
        mass_properties = deflector.deflector(deflector_index=0).mass_properties
        assert (mass_properties["halo_mass"] > 1e12) and (
            mass_properties["halo_mass"] < 3e15
        )
        assert (mass_properties["concentration"] > 1) and (
            mass_properties["concentration"] < 15
        )
        assert (len(members) >= 1) and (len(members) < 100)


if __name__ == "__main__":
    pytest.main()
