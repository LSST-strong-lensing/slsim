from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.SourcePopulation.point_plus_extended_sources import (
    PointPlusExtendedSources,
)
import os
from astropy.table import Table
import pytest


class TestPointPlusExtendedSources(object):
    def setup_method(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        new_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # new_path = self.path.replace("test_Sources", "/")
        self.source_list = Table.read(
            os.path.join(new_path, "TestData/point_source_catalog_test.fits"),
            format="fits",
        )
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        kwargs = {
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "agn_driving_variability_model": None,
            "agn_driving_kwargs_variability": None,
            "lightcurve_time": None,
        }
        self.pe_source = PointPlusExtendedSources(
            point_plus_extended_sources_list=self.source_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            point_source_type="quasar",
            extended_source_type="single_sersic",
            point_source_kwargs=kwargs,
        )

    def test_source_number(self):
        number = self.pe_source.source_number
        assert number > 0

    def test_draw_source(self):
        point_plus_extended_sources = self.pe_source.draw_source()
        assert isinstance(point_plus_extended_sources, object)
        assert point_plus_extended_sources.redshift > 0
        assert point_plus_extended_sources.source_type == "point_plus_extended"

        point_plus_extended_sources2 = self.pe_source.draw_source(z_max=-1)
        assert point_plus_extended_sources2 is None


if __name__ == "__main__":
    pytest.main()
