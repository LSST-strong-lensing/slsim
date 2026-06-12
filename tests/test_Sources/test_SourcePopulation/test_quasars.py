from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.SourcePopulation.point_sources import PointSources
from slsim.Sources.SourceCatalogues.QuasarCatalog.simple_quasar import (
    quasar_catalog_simple,
)
import pytest


@pytest.fixture
def Quasar_class():
    sky_area = Quantity(value=0.1, unit="deg2")
    kwargs_quasars = {
        "num_quasars": 5000,
        "z_min": 0.1,
        "z_max": 5,
        "m_min": 17,
        "m_max": 25,
    }
    quasar_list = quasar_catalog_simple(**kwargs_quasars)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kwargs = {
        "variability_model": "light_curve",
        "kwargs_variability": None,
        "agn_driving_variability_model": None,
        "agn_driving_kwargs_variability": None,
        "lightcurve_time": None,
    }
    return PointSources(
        point_source_list=quasar_list,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut={},
        point_source_type="quasar",
        point_source_kwargs=kwargs,
    )


def test_source_number(Quasar_class):
    number = Quasar_class.source_number
    assert number > 0


def test_draw_source(Quasar_class):
    quasar = Quasar_class.draw_source()
    assert isinstance(quasar, object)
    assert quasar.redshift > 0


def test_draw_source_with_redshift_limits(Quasar_class):
    # Test a valid redshift range where we expect to find sources
    quasar = Quasar_class.draw_source(z_min=1.0, z_max=2.0)
    assert quasar is not None
    assert 1.0 < quasar.redshift < 2.0

    # Test an empty redshift range where we expect to find no sources
    quasar_none = Quasar_class.draw_source(z_min=10.0, z_max=12.0)
    assert quasar_none is None

    # Test with only z_min provided
    quasar_zmin_only = Quasar_class.draw_source(z_min=1.0)
    assert quasar_zmin_only is not None
    assert quasar_zmin_only.redshift > 1.0

    # Test with only z_max provided
    quasar_zmax_only = Quasar_class.draw_source(z_max=3.0)
    assert quasar_zmax_only is not None
    assert 0 <= quasar_zmax_only.redshift < 3.0


def test_draw_source_with_index(Quasar_class):
    # Test fetching a specific source by its index
    quasar = Quasar_class.draw_source(point_source_index=0)
    assert isinstance(quasar, object)
    assert quasar.redshift > 0


def test_source_number_selected(Quasar_class):
    num = Quasar_class.source_number_selected
    assert num > 0


def test_variability_model(Quasar_class):
    kwargs_variab = Quasar_class.point_source_kwargs["variability_model"]
    assert kwargs_variab == "light_curve"


def test_kwarg_variability(Quasar_class):
    kwargs_variab = Quasar_class.point_source_kwargs["kwargs_variability"]
    assert kwargs_variab is None


if __name__ == "__main__":
    pytest.main()
