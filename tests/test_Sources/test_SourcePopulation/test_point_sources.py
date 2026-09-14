from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.SourcePopulation.point_sources import PointSources
from slsim.Sources.SourceCatalogues.QuasarCatalog.simple_quasar import (
    quasar_catalog_simple,
)
from slsim.Sources.source import Source


class TestPointSources(object):

    def setup_method(self):

        sky_area = Quantity(value=0.1, unit="deg2")
        kwargs_quasars = {
            "num_quasars": 5000,
            "z_min": 0.1,
            "z_max": 5,
            "m_min": 17,
            "m_max": 25,
        }
        self.quasar_list = quasar_catalog_simple(**kwargs_quasars)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        kwargs = {
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "agn_driving_variability_model": None,
            "agn_driving_kwargs_variability": None,
            "lightcurve_time": None,
        }
        self.point_source = PointSources(
            point_source_list=self.quasar_list,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut={},
            point_source_type="quasar",
            joint_point_source_kwargs=kwargs,
        )

    def test_source_number(self):
        number = self.point_source.source_number
        assert number > 0

    def test_draw_source(self):
        quasar = self.point_source.draw_source()
        assert isinstance(quasar, object)
        assert quasar.redshift > 0

    def test_point_source_init(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        point_sources = PointSources(
            point_source_list=self.quasar_list,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut=None,
            point_source_type="quasar",
            joint_point_source_kwargs=None,
        )
        point_source = point_sources.draw_source()
        assert isinstance(point_source, Source)

    def test_catalog_overrides_joint_kwargs_on_collision(self):
        """A catalog column should override a same-named joint kwarg, and
        should not crash."""
        sky_area = Quantity(value=0.1, unit="deg2")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        catalog = self.quasar_list.copy()
        catalog["black_hole_mass_exponent"] = 9.0
        kwargs = {
            "variability_model": "light_curve",
            "kwargs_variability": None,
            "black_hole_mass_exponent": 7.0,
        }
        point_sources = PointSources(
            point_source_list=catalog,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut={},
            point_source_type="quasar",
            joint_point_source_kwargs=kwargs,
        )
        source = point_sources.draw_source()
        assert source.point_source._black_hole_mass_exponent == 9.0
