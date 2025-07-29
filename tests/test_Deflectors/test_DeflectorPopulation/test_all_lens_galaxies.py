from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.DeflectorPopulation.all_lens_galaxies import (
    AllLensGalaxies,
    fill_table,
)
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Table
import numpy as np
import pytest
import copy


def galaxy_list():
    sky_area = Quantity(value=0.001, unit="deg2")
    pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
    return pipeline.red_galaxies, pipeline.blue_galaxies


galaxies = galaxy_list()


@pytest.fixture
def all_lens_galaxies():
    galaxy_list = copy.copy(galaxies)
    red_galaxies = galaxy_list[0]
    blue_galaxies = galaxy_list[1]
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    return AllLensGalaxies(
        red_galaxies,
        blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


def test_deflector_number_draw_deflector(all_lens_galaxies):
    galaxy_pop = all_lens_galaxies
    num_deflectors = galaxy_pop.deflector_number()
    deflector = galaxy_pop.draw_deflector()
    assert deflector.redshift != 0
    assert num_deflectors >= 0


def test_fill_table():
    mock_galaxy_list = copy.copy(galaxies)[0]
    filled_table = fill_table(mock_galaxy_list)
    assert isinstance(filled_table, Table)


def test_vel_disp_abundance_matching():
    mock_galaxy_list = copy.copy(galaxies)[0]
    sky_area = Quantity(value=0.05, unit="deg2")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    f_vel_disp = vel_disp_abundance_matching(
        mock_galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
    )

    assert callable(f_vel_disp)
    stellar_mass = 10 ** np.random.uniform(9, 12, 10)
    vel_disp = f_vel_disp(np.log10(stellar_mass))
    assert isinstance(vel_disp, np.ndarray)


def test_all_lens_galaxies_2():
    galaxy_list = copy.copy(galaxies)
    red_galaxies = copy.copy(galaxy_list[0])
    blue_galaxies = copy.copy(galaxy_list[1])
    red_galaxies2 = copy.copy(galaxy_list[0])
    blue_galaxies2 = copy.copy(galaxy_list[1])
    red_galaxies3 = copy.copy(galaxy_list[0])
    blue_galaxies3 = copy.copy(galaxy_list[1])
    red_galaxies4 = copy.copy(galaxy_list[0])
    blue_galaxies4 = copy.copy(galaxy_list[1])
    red_galaxies5 = copy.copy(galaxy_list[0])
    blue_galaxies5 = copy.copy(galaxy_list[1])
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    galaxy_class1 = AllLensGalaxies(
        red_galaxies,
        blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl=2.05,
    )
    galaxy_class2 = AllLensGalaxies(
        red_galaxies2,
        blue_galaxies2,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl={"mean": 2.1, "std_dev": 0.16},
    )
    galaxy_class3 = AllLensGalaxies(
        red_galaxies3,
        blue_galaxies3,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl={"gamma_min": 1.95, "gamma_max": 2.26},
    )
    assert galaxy_class1.draw_deflector().halo_properties["gamma_pl"] == 2.05
    assert 1.6 <= galaxy_class2.draw_deflector().halo_properties["gamma_pl"] <= 2.6
    assert 1.95 <= galaxy_class3.draw_deflector().halo_properties["gamma_pl"] <= 2.26
    with pytest.raises(ValueError):
        AllLensGalaxies(
            red_galaxies4,
            blue_galaxies4,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl={"gamma_mi": 1.95, "gamma_ma": 2.26},
        )
    with pytest.raises(ValueError):
        AllLensGalaxies(
            red_galaxies5,
            blue_galaxies5,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl=[2.1, 0.16],
        )


if __name__ == "__main__":
    pytest.main()
