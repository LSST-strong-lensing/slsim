import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from slsim.LOS.los_individual import LOSIndividual
import slsim.Sources as sources
import slsim.Deflectors as deflectors
import slsim.Pipelines as pipelines
from slsim.FalsePositives.false_positive import FalsePositive
from astropy.units import Quantity

sky_area = Quantity(value=0.01, unit="deg2")
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None,
    sky_area=sky_area,
    filters=None,
)
kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}


def test_false_positive():
    # Mock objects for source_class and deflector_class

    # Initialize a cosmology instance
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lens_galaxies = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=0.1,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    kwargs = {"extended_source_type": "single_sersic"}
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        **kwargs
    )
    single_deflector = lens_galaxies.draw_deflector()
    single_source1 = source_galaxies.draw_source()
    single_source2 = source_galaxies.draw_source()
    lens = single_deflector
    source = single_source1
    source2 = single_source2
    source_list = [source, source2]
    # LOS configuration
    los_class = LOSIndividual()

    # Create an instance of FalsePositive
    false_positive_instance_1 = FalsePositive(
        source_class=source,
        deflector_class=lens,
        cosmo=cosmo,
    )
    false_positive_instance_2 = FalsePositive(
        source_class=source_list,
        deflector_class=lens,
        cosmo=cosmo,
        los_class=los_class,
    )
    required_keys = {
        "magnitude",
        "R_sersic",
        "n_sersic",
        "e1",
        "e2",
        "center_x",
        "center_y",
    }
    assert false_positive_instance_1.source_number == 1
    assert false_positive_instance_2.source_number == 2
    assert (
        false_positive_instance_1.lenstronomy_kwargs("i")[0]["lens_light_model_list"][0]
        == "SERSIC_ELLIPSE"
    )
    assert (
        len(
            false_positive_instance_2.lenstronomy_kwargs("i")[0][
                "lens_light_model_list"
            ]
        )
        == 3
    )
    assert np.all(
        false_positive_instance_1.lenstronomy_kwargs("i")[0]["lens_model_list"]
        == ["SIE", "SHEAR", "CONVERGENCE"]
    )
    assert (
        len(false_positive_instance_2.lenstronomy_kwargs("i")[1]["kwargs_lens_light"])
        == 3
    )
    assert len(false_positive_instance_2.deflector_position) == 2
    assert false_positive_instance_2.deflector_redshift[0] == single_deflector.redshift
    assert false_positive_instance_1.source_redshift_list[0] == single_source1.redshift
    assert np.all(false_positive_instance_2.source_redshift_list) == np.all(
        np.array([single_source1.redshift, single_source2.redshift])
    )
    assert false_positive_instance_1.external_convergence < 0.1
    assert false_positive_instance_1.external_shear < 0.2
    assert false_positive_instance_1.einstein_radius[0] < 2.5
    assert false_positive_instance_1.deflector_magnitude(
        band="i"
    ) == single_deflector.magnitude(band="i")
    assert false_positive_instance_1.extended_source_magnitude(
        band="i"
    ) == single_source1.extended_source_magnitude(band="i")
    assert len(false_positive_instance_1.deflector_ellipticity()) == 4
    assert (
        false_positive_instance_1.deflector_stellar_mass()
        == single_deflector.stellar_mass
    )
    assert (
        set(
            false_positive_instance_1.deflector_light_model_lenstronomy(band="i")[1][
                0
            ].keys()
        )
        == required_keys
    )


if __name__ == "__main__":
    pytest.main()
