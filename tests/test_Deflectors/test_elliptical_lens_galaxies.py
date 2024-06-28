from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.elliptical_lens_galaxies import (
    EllipticalLensGalaxies, vel_disp_from_m_star
)
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Table
import numpy as np
import pytest


def galaxy_list():
    sky_area = Quantity(value=0.005, unit="deg2")
    pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
    return pipeline.red_galaxies


@pytest.fixture
def elliptical_lens_galaxies():
    red_galaxies = galaxy_list()
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.005, unit="deg2")
    return EllipticalLensGalaxies(
        red_galaxies,
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
    assert deflector["vel_disp"] == vel_disp_from_m_star(deflector["stellar_mass"])


if __name__ == "__main__":
    pytest.main()
