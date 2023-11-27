import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.lens_pop import LensPop, draw_test_area
import numpy as np


@pytest.fixture
def gg_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    return LensPop(sky_area=sky_area, cosmo=cosmo)


def test_pes_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    pes_lens_pop = LensPop(
        deflector_type="all-galaxies",
        source_type="quasar_plus_galaxies",
        variability_model="sinusoidal",
        kwargs_variability={"amp", "freq"},
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"


def test_num_lenses_and_sources(gg_lens_pop_instance):
    num_lenses = gg_lens_pop_instance.deflector_number()
    num_sources = gg_lens_pop_instance.source_number()

    assert 100 <= num_lenses <= 6600, "Expected num_lenses to be between 5800 and 6600,"
    f"but got {num_lenses}"
    assert (
        100000 <= num_sources <= 500000
    ), "Expected num_sources to be between 1090000 and"
    f"1110000, but got {num_sources}"
    # assert 1 == 0


def test_num_sources_tested_and_test_area(gg_lens_pop_instance):
    lens = gg_lens_pop_instance._lens_galaxies.draw_deflector()
    test_area = draw_test_area(deflector=lens)
    assert (
        0.01 < test_area < 100 * np.pi
    ), "Expected test_area to be between 0.1 and 100*pi,"
    f"but got {test_area}"
    num_sources_range = gg_lens_pop_instance.get_num_sources_tested(testarea=test_area)
    assert (
        0 <= num_sources_range <= 50
    ), "Expected num_sources_range to be between 0 and 50,"
    f"but got {num_sources_range}"


def test_draw_population(gg_lens_pop_instance):
    kwargs_lens_cuts = {}
    gg_lens_population = gg_lens_pop_instance.draw_population(kwargs_lens_cuts)
    assert isinstance(gg_lens_population, list)


if __name__ == "__main__":
    pytest.main()
