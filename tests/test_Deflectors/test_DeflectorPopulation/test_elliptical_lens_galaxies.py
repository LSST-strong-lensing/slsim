from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.DeflectorPopulation.elliptical_lens_galaxies import (
    EllipticalLensGalaxies,
)
from slsim.Util.param_util import vel_disp_from_m_star
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
import copy
import pytest


def galaxy_list():
    sky_area = Quantity(value=0.001, unit="deg2")
    pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
    red_gal = pipeline.red_galaxies
    return red_gal


galaxies = galaxy_list()


@pytest.fixture
def elliptical_lens_galaxies():
    red_galaxies = copy.copy(galaxies)
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")
    return EllipticalLensGalaxies(
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


def test_deflector_number_draw_deflector(elliptical_lens_galaxies):
    galaxy_pop = elliptical_lens_galaxies
    num_deflectors = galaxy_pop.deflector_number()
    deflector = galaxy_pop.draw_deflector()
    assert deflector.redshift != 0
    assert num_deflectors >= 0


def test_vel_disp_from_m_star():
    assert vel_disp_from_m_star(0) == 0


def test_elliptical_lens_galaxies_2():
    red_galaxies = copy.copy(galaxies)
    red_galaxies2 = copy.copy(galaxies)
    red_galaxies3 = copy.copy(galaxies)
    red_galaxies4 = copy.copy(galaxies)
    red_galaxies5 = copy.copy(galaxies)
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")
    galaxy_class1 = EllipticalLensGalaxies(
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl=2.15,
    )
    galaxy_class2 = EllipticalLensGalaxies(
        red_galaxies2,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl={"mean": 2.0, "std_dev": 0.16},
    )
    galaxy_class3 = EllipticalLensGalaxies(
        red_galaxies3,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl={"gamma_min": 1.8, "gamma_max": 2.3},
    )
    assert galaxy_class1.draw_deflector().halo_properties["gamma_pl"] == 2.15
    assert 1.5 <= galaxy_class2.draw_deflector().halo_properties["gamma_pl"] <= 2.5
    assert 1.8 <= galaxy_class3.draw_deflector().halo_properties["gamma_pl"] <= 2.3
    with pytest.raises(ValueError):
        EllipticalLensGalaxies(
            red_galaxies4,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl={"gamma_mi": 1.8, "gamma_ma": 2.3},
        )
    with pytest.raises(ValueError):
        EllipticalLensGalaxies(
            red_galaxies5,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl=[2.0, 0.16],
        )


if __name__ == "__main__":
    pytest.main()
