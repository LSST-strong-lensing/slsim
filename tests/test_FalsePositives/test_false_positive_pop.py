import pytest
from astropy.cosmology import FlatLambdaCDM
import slsim.Sources as sources
import slsim.Deflectors as deflectors
import slsim.Pipelines as pipelines
from slsim.FalsePositives.false_positive_pop import FalsePositivePop
from astropy.units import Quantity

sky_area = Quantity(value=0.01, unit="deg2")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None,
    sky_area=sky_area,
    filters=None,
)
kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}
red_galaxy_list = galaxy_simulation_pipeline.red_galaxies
blue_galaxy_list = galaxy_simulation_pipeline.blue_galaxies
lens_galaxies = deflectors.EllipticalLensGalaxies(
    galaxy_list=red_galaxy_list,
    kwargs_cut=kwargs_deflector_cut,
    kwargs_mass2light=0.1,
    cosmo=cosmo,
    sky_area=sky_area,
)
kwargs = {"extended_source_type": "single_sersic"}
source_galaxies = sources.Galaxies(
    galaxy_list=blue_galaxy_list,
    kwargs_cut=kwargs_source_cut,
    cosmo=cosmo,
    sky_area=sky_area,
    catalog_type="skypy",
    **kwargs
)


def test_draw_false_positive_single():
    fp_pop1 = FalsePositivePop(
        elliptical_galaxy_population=lens_galaxies,
        blue_galaxy_population=source_galaxies,
        cosmo=cosmo,
        source_number_choice=[1],
    )
    draw_fp1 = fp_pop1.draw_false_positive()
    draw_deflector = fp_pop1.draw_deflector()
    draw_source = fp_pop1.draw_sources(z_max=draw_deflector[1])
    assert isinstance(draw_fp1, object)
    assert isinstance(draw_deflector[0], object)
    assert draw_deflector[1] == draw_deflector[0].redshift + 0.002
    assert isinstance(draw_source, object)


def test_draw_false_positive_multiple():
    fp_pop2 = FalsePositivePop(
        elliptical_galaxy_population=lens_galaxies,
        blue_galaxy_population=source_galaxies,
        cosmo=cosmo,
        source_number_choice=[2],
    )
    draw_fp2 = fp_pop2.draw_false_positive(number=2)
    assert isinstance(draw_fp2, list)


if __name__ == "__main__":
    pytest.main()
