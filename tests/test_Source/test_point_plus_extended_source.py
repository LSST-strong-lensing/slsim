from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.point_plus_extended_source import PointPlusExtendedSource
import os
from astropy.table import Table
import pytest


class TestPointPlusExtendedSource(object):
    def setup_method(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        self.path = os.path.dirname(__file__)
        new_path = self.path.replace("test_Source", "/")
        self.source_list = Table.read(
            os.path.join(new_path, "TestData/point_source_catalog_test.fits"),
            format="fits",
        )
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.pe_source = PointPlusExtendedSource(
            point_plus_extended_source_list=self.source_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
        )

    def test_source_number(self):
        number = self.pe_source.source_number()
        assert number > 0

    def test_draw_source(self):
        point_plus_extended_source = self.pe_source.draw_source()
        assert len(point_plus_extended_source) > 0


if __name__ == "__main__":
    pytest.main()
