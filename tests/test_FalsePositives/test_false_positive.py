import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from slsim.LOS.los_individual import LOSIndividual
import slsim.Sources as sources
import slsim.Deflectors as deflectors
import slsim.Pipelines as pipelines
from slsim.FalsePositives.false_positive import FalsePositive
from astropy.units import Quantity
from astropy.table import Table

sky_area = Quantity(value=0.01, unit="deg2")
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None,
    sky_area=sky_area,
    filters=None,
)
kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}


@pytest.fixture
def fp_test_setup():
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
    path = os.path.dirname(__file__)
    loaded_qso_host_catalog = Table.read(
        os.path.join(path, "../TestData/qso_host_catalog.fits")
    )
    source_quasars = sources.PointPlusExtendedSources(
        point_plus_extended_sources_list=loaded_qso_host_catalog,
        cosmo=cosmo,
        kwargs_cut={"band": "g", "band_max": 28, "z_min": 2, "z_max": 5.0},
        sky_area=Quantity(12.0, unit="deg2"),
        point_source_type="quasar",
        point_source_kwargs={},
        extended_source_type="double_sersic",
        extendedsource_kwargs={},
    )
    return cosmo, lens_galaxies, source_galaxies, source_quasars


def test_false_positive(fp_test_setup):
    cosmo, lens_galaxies, source_galaxies, source_quasars = fp_test_setup

    single_deflector = lens_galaxies.draw_deflector()
    single_source1 = source_galaxies.draw_source()
    single_source2 = source_galaxies.draw_source()
    single_source_quasar = source_quasars.draw_source()
    lens = single_deflector
    source = single_source1
    source2 = single_source2
    source_list = [source, source2, single_source_quasar]
    los_class = LOSIndividual()

    # Create instances of FalsePositive
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
    assert false_positive_instance_2.source_number == 3

    kw_model_1, _ = false_positive_instance_1.lenstronomy_kwargs("i")
    kw_model_2, _ = false_positive_instance_2.lenstronomy_kwargs("i")

    assert kw_model_1["lens_light_model_list"][0] == "SERSIC_ELLIPSE"
    assert np.all(kw_model_1["lens_model_list"] == ["SIE", "SHEAR", "CONVERGENCE"])

    assert (
        len(kw_model_2["lens_light_model_list"]) == 5
    )  # 1 deflector + 2 extended sources + (2 extended sources from host double sersic) = 5 total lens light models

    assert false_positive_instance_1.source(0).extended_source.lensed is False

    # instance 2 has extended sources at index 0 and 1
    assert false_positive_instance_2.source(0).extended_source.lensed is False
    assert false_positive_instance_2.source(1).extended_source.lensed is False

    # instance 2 has a Point+Extended source (Quasar+double-sersic host) at index 2
    assert false_positive_instance_2.source(2).extended_source is not None
    assert false_positive_instance_2.source(2).point_source is not None
    assert false_positive_instance_2.source(2).extended_source.lensed is False
    assert false_positive_instance_2.source(2).point_source.lensed is False

    assert len(false_positive_instance_2.deflector_position) == 2
    assert false_positive_instance_2.deflector_redshift[0] == single_deflector.redshift
    assert false_positive_instance_1.source_redshift_list[0] == single_source1.redshift
    assert np.all(false_positive_instance_2.source_redshift_list) == np.all(
        np.array(
            [
                single_source1.redshift,
                single_source2.redshift,
                single_source_quasar.redshift,
            ]
        )
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


def test_false_positive_toggles(fp_test_setup):
    """Test the include_deflector_light and field_galaxies additions."""
    cosmo, lens_galaxies, source_galaxies, _ = fp_test_setup

    lens = lens_galaxies.draw_deflector()
    source = source_galaxies.draw_source()
    mock_field_galaxy = source_galaxies.draw_source()

    # Test 1: Exclude deflector light
    fp_no_def_light = FalsePositive(
        source_class=source,
        deflector_class=lens,
        cosmo=cosmo,
        include_deflector_light=False,
    )
    kwargs_model_no_def, _ = fp_no_def_light.lenstronomy_kwargs("i")
    # Only the single source light model should be present
    assert len(kwargs_model_no_def["lens_light_model_list"]) == 1

    # Test 2: Include field galaxies
    fp_with_field = FalsePositive(
        source_class=source,
        deflector_class=lens,
        cosmo=cosmo,
        include_deflector_light=True,
        field_galaxies=[mock_field_galaxy],
    )
    kwargs_model_field, _ = fp_with_field.lenstronomy_kwargs("i")
    # 1 source + 1 deflector + 1 field galaxy = 3 light models
    assert len(kwargs_model_field["lens_light_model_list"]) == 3


def test_false_positive_overridden_physics_methods(fp_test_setup):
    """Tests the overridden methods in the FalsePositive class that enforce an
    'unlensed' physical scenario."""
    cosmo, lens_galaxies, source_galaxies, _ = fp_test_setup

    lens = lens_galaxies.draw_deflector()
    source = source_galaxies.draw_source()

    fp_instance = FalsePositive(
        source_class=source,
        deflector_class=lens,
        cosmo=cosmo,
    )

    # 1. Test _image_position_from_source
    x_source_test, y_source_test = 1.5, -0.75
    x_img, y_img = fp_instance._image_position_from_source(
        x_source=x_source_test, y_source=y_source_test, source_index=0
    )
    assert np.array_equal(x_img, np.array([x_source_test]))
    assert np.array_equal(y_img, np.array([y_source_test]))

    # 2. Test _point_source_magnification
    mag = fp_instance._point_source_magnification(source_index=0)
    assert np.array_equal(mag, np.array([1.0]))

    # 3. Test _point_source_arrival_times
    arrival_times = fp_instance._point_source_arrival_times(source_index=0)
    assert np.array_equal(arrival_times, np.array([0.0]))


if __name__ == "__main__":
    pytest.main()
