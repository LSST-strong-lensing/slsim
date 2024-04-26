import pytest
from astropy.cosmology import FlatLambdaCDM
from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline
from astropy.units import Quantity
from slsim.Deflectors.compound_lens_halos_galaxies import CompoundLensHalosGalaxies

# Assuming other imports are already defined, we continue from here.


def galaxy_list(sky_area, cosmo):
    pipeline = SLHammocksPipeline(
        slhammocks_config=None, sky_area=sky_area, cosmo=cosmo
    )
    return pipeline._pipeline


@pytest.fixture
def compound_lens_halos_galaxies():
    sky_area = Quantity(value=0.05, unit="deg2")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
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
    halo_galaxy_pop = compound_lens_halos_galaxies  # Example subset
    num_deflectors = halo_galaxy_pop.deflector_number()
    deflector = halo_galaxy_pop.draw_deflector()
    assert deflector["z"] != 0
    assert num_deflectors >= 0
    assert deflector["vel_disp"] != -1
    assert deflector["e1_light"] != -1
    assert deflector["e2_light"] != -1
    assert deflector["e1_mass"] != -1
    assert deflector["e2_mass"] != -1


# The following decorator and function are needed to run the tests with pytest
if __name__ == "__main__":
    pytest.main()
