import numpy as np

from astropy import cosmology
from astropy.units import Quantity
from astropy.table import Table

from slsim.Sources.Events.event_lightcone import EventLightcone
from slsim.Sources.SourceCatalogues.BNSCatalog.bns_catalog import BNSCatalog
from slsim.Sources.SourcePopulation.point_sources import PointSources


class TestBNSCatalog:
    def setup_method(self):
        self.cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.sky_area = Quantity(value=0.1, unit="deg2")
        self.lightcurve_time = np.linspace(0.1, 10, 50)

    def _bns_data(self, monkeypatch):
        test_redshifts = np.array([0.2, 0.5, 0.8])

        monkeypatch.setattr(
            EventLightcone,
            "event_sample",
            lambda self: test_redshifts,
        )

        catalog_class = BNSCatalog(
            cosmo=self.cosmo,
            sky_area=self.sky_area,
            band_list=["i", "r"],
            lightcurve_time=self.lightcurve_time,
        )

        bns_data = catalog_class.bns_catalog()

        return catalog_class, bns_data, test_redshifts

    def test_bns_catalog(self, monkeypatch):
        _, bns_table, test_redshifts = self._bns_data(monkeypatch)

        assert isinstance(bns_table, Table)
        assert len(bns_table) == 3
        assert "z" in bns_table.colnames
        assert "lightcurve_time" in bns_table.colnames
        assert "model_name" in bns_table.colnames
        assert "variability_model" in bns_table.colnames
        assert "mag_zpsys" in bns_table.colnames

        np.testing.assert_array_equal(
            bns_table["z"],
            test_redshifts,
        )

    def test_bns_catalog_to_source(self, monkeypatch):
        catalog_class, bns_data, test_redshifts = self._bns_data(monkeypatch)

        joint_point_source_kwargs = {
            "variability_model": "light_curve",
            "kwargs_variability": ["bns_lightcurve", "i", "r"],
            "lightcurve_time": self.lightcurve_time,
            "model_name": "mosfit_kilonova",
            "mag_zpsys": "AB",
            "modeldir": None,
            "kwargs_kilonova": catalog_class._kwargs_kilonova,
        }

        bns_population = PointSources(
            point_source_list=bns_data,
            cosmo=self.cosmo,
            sky_area=self.sky_area,
            kwargs_cut={},
            point_source_type="kilonova",
            joint_point_source_kwargs=joint_point_source_kwargs,
        )

        source = bns_population.draw_source()

        assert source.redshift in test_redshifts
        assert source.source_type == "point_source"

        mag = source.point_source_magnitude("i")
        assert np.isfinite(mag)

    def test_default_kwargs_kilonova(self):
        catalog_class = BNSCatalog(
            cosmo=self.cosmo,
            sky_area=self.sky_area,
            band_list=["i", "r"],
            lightcurve_time=self.lightcurve_time,
        )

        assert catalog_class._kwargs_kilonova["mej_1"] == 0.01
        assert catalog_class._kwargs_kilonova["mej_2"] == 0.02
        assert catalog_class._kwargs_kilonova["mej_3"] == 0.03
        assert catalog_class._kwargs_kilonova["vej_1"] == 0.1
        assert catalog_class._kwargs_kilonova["vej_2"] == 0.2
        assert catalog_class._kwargs_kilonova["vej_3"] == 0.3
        assert catalog_class._kwargs_kilonova["kappa_1"] == 0.5
        assert catalog_class._kwargs_kilonova["kappa_2"] == 3.0
        assert catalog_class._kwargs_kilonova["kappa_3"] == 10.0
        assert catalog_class._kwargs_kilonova["temperature_floor_1"] == 5000
        assert catalog_class._kwargs_kilonova["temperature_floor_2"] == 4000
        assert catalog_class._kwargs_kilonova["temperature_floor_3"] == 3000
        assert catalog_class._kwargs_kilonova["kappa_gamma"] == 10

    def test_custom_kwargs_kilonova(self):
        catalog_class = BNSCatalog(
            cosmo=self.cosmo,
            sky_area=self.sky_area,
            band_list=["i", "r"],
            lightcurve_time=self.lightcurve_time,
            kwargs_kilonova={"mej_1": 0.05},
        )

        assert catalog_class._kwargs_kilonova["mej_1"] == 0.05
        assert catalog_class._kwargs_kilonova["mej_2"] == 0.02
