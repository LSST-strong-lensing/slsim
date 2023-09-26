from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from sim_pipeline.Sources.quasars import Quasar
from sim_pipeline.Sources.quasar_catalog.simple_quasar import quasar_catalog
import pytest


class test_quasar(object):
    def setup_method(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        self.quasar_list = quasar_catalog(50000, 0.1, 5, 17, 23)
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.quasar = Quasar(
            galaxy_list=self.quasar_list,
            kwargs_cut=None,
            cosmo=self.cosmo,
            sky_area=sky_area,
        )

    def test_source_number(self):
        number = self.quasar.source_number()
        assert number > 0

    def test_draw_source(self):
        quasar = self.quasar.draw_source()
        assert len(quasar) > 0

if __name__ == "__main__":
    pytest.main()
