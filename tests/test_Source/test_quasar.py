from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.quasars import Quasars
from slsim.Sources.QuasarCatalog.simple_quasar import quasar_catalog_simple
import pytest


@pytest.fixture
def Quasar_class():
    sky_area = Quantity(value=0.1, unit="deg2")
    kwargs_quasars = {
        "num_quasars": 50000,
        "z_min": 0.1,
        "z_max": 5,
        "m_min": 17,
        "m_max": 25,
    }
    quasar_list = quasar_catalog_simple(**kwargs_quasars)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    return Quasars(
        quasar_list=quasar_list,
        cosmo=cosmo,
        sky_area=sky_area,
        variability_model="sinusoidal",
        kwargs_variability_model={"amp", "freq"},
    )


def test_source_number(Quasar_class):
    number = Quasar_class.source_number()
    assert number > 0


def test_draw_source(Quasar_class):
    quasar = Quasar_class.draw_source()
    assert len(quasar) > 0


def test_variability_model(Quasar_class):
    kwargs_variab = Quasar_class.variability_model
    assert kwargs_variab == "sinusoidal"


def test_kwarg_variability(Quasar_class):
    kwargs_variab = Quasar_class.kwargs_variability
    assert kwargs_variab == {"amp", "freq"}


if __name__ == "__main__":
    pytest.main()
