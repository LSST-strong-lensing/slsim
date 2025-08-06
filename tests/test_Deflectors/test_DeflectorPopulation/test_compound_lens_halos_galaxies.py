import pytest
from astropy.cosmology import FlatLambdaCDM
from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline
from astropy.units import Quantity
from slsim.Deflectors.DeflectorPopulation.compound_lens_halos_galaxies import (
    CompoundLensHalosGalaxies,
)
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

# Assuming other imports are already defined, we continue from here.


def galaxy_list(sky_area, cosmo):
    pipeline = SLHammocksPipeline(
        slhammocks_config=None, sky_area=sky_area, cosmo=cosmo, z_min=0.01, z_max=5
    )
    return pipeline.halo_galaxies


@pytest.fixture
def compound_lens_halos_galaxies():
    sky_area = Quantity(value=0.05, unit="deg2")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.725)
    halos_galaxies = galaxy_list(sky_area, cosmo)
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}

    return CompoundLensHalosGalaxies(
        halos_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


def test_deflector_number_draw_deflector(compound_lens_halos_galaxies):
    # Mocking the deflector_cut function to return a subset of the galaxy list
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.725)
    halo_galaxy_pop = compound_lens_halos_galaxies  # Example subset
    num_deflectors = halo_galaxy_pop.deflector_number()
    deflector = halo_galaxy_pop.draw_deflector()
    lens_cosmo = LensCosmo(
        z_lens=float(deflector.redshift),
        z_source=float(deflector.redshift + 0.5),
        cosmo=cosmo,
    )
    light_lenstronomy = deflector.light_model_lenstronomy()
    mass_lenstronomy = deflector.mass_model_lenstronomy(lens_cosmo)
    expected_mass_model = ["NFW_ELLIPSE_CSE", "HERNQUIST_ELLIPSE_CSE"]
    expected_light_model = ["HERNQUIST_ELLIPSE"]
    assert deflector.redshift > 0
    assert num_deflectors >= 0
    assert deflector.velocity_dispersion(cosmo=cosmo) > 0
    assert mass_lenstronomy[0][0] in expected_mass_model
    assert mass_lenstronomy[0][1] in expected_mass_model
    assert light_lenstronomy[0][0] in expected_light_model


# The following decorator and function are needed to run the tests with pytest
if __name__ == "__main__":
    pytest.main()
