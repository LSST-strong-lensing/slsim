from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.all_lens_galaxies import (
    AllLensGalaxies,
    fill_table,
    vel_disp_abundance_matching,
)
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Table
import numpy as np
import pytest


def galaxy_list():
    sky_area = Quantity(value=0.05, unit="deg2")
    pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
    return pipeline.red_galaxies, pipeline.blue_galaxies


@pytest.fixture
def all_lens_galaxies():
    red_galaxies = galaxy_list()[0]
    blue_galaxies = galaxy_list()[1]
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
    assert deflector["z"] != 0
    assert num_deflectors >= 0


def test_fill_table():
    mock_galaxy_list = galaxy_list()[0]
    filled_table = fill_table(mock_galaxy_list)
    assert isinstance(filled_table, Table)


def test_vel_disp_abundance_matching():
    mock_galaxy_list = galaxy_list()[0]
    sky_area = Quantity(value=0.05, unit="deg2")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    f_vel_disp = vel_disp_abundance_matching(
        mock_galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
    )

    assert callable(f_vel_disp)
    stellar_mass = 10 ** np.random.uniform(9, 12, 10)
    vel_disp = f_vel_disp(np.log10(stellar_mass))
    assert isinstance(vel_disp, np.ndarray)


if __name__ == "__main__":
    pytest.main()
