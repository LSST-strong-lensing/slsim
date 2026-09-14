from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.DeflectorPopulation.galaxy_deflectors import (
    GalaxyDeflectors,
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
    return GalaxyDeflectors(
        red_galaxies,
        blue_galaxy_list=blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
    )


@pytest.fixture
def red_galaxies():
    galaxy_list = copy.copy(galaxies)
    red_galaxies = galaxy_list[0]
    # blue_galaxies = galaxy_list[1]
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    return GalaxyDeflectors(
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
    assert deflector.redshift != 0
    assert num_deflectors >= 0


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
    galaxy_class1 = GalaxyDeflectors(
        red_galaxies,
        blue_galaxy_list=blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl=2.05,
    )
    galaxy_class2 = GalaxyDeflectors(
        red_galaxies2,
        blue_galaxy_list=blue_galaxies2,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl={"mean": 2.1, "std_dev": 0.16},
    )
    galaxy_class3 = GalaxyDeflectors(
        red_galaxies3,
        blue_galaxy_list=blue_galaxies3,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl={"gamma_min": 1.95, "gamma_max": 2.26},
    )
    assert galaxy_class1.draw_deflector().mass_properties["gamma_pl"] == 2.05
    assert 1.6 <= galaxy_class2.draw_deflector().mass_properties["gamma_pl"] <= 2.6
    assert 1.95 <= galaxy_class3.draw_deflector().mass_properties["gamma_pl"] <= 2.26
    with pytest.raises(ValueError):
        deflectors = GalaxyDeflectors(
            red_galaxies4,
            blue_galaxy_list=blue_galaxies4,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl={"gamma_mi": 1.95, "gamma_ma": 2.26},
        )
        deflectors.draw_deflector()
    with pytest.raises(ValueError):
        GalaxyDeflectors(
            red_galaxies5,
            blue_galaxy_list=blue_galaxies5,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light=kwargs_mass2light,
            cosmo=cosmo,
            sky_area=sky_area,
            gamma_pl=[2.1, 0.16],
        )
        deflectors.draw_deflector()


def test_elliptical_galaxies():
    galaxy_list = copy.copy(galaxies)
    red_galaxies = copy.copy(galaxy_list[0])

    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    galaxy_class1 = GalaxyDeflectors(
        red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        gamma_pl=2.05,
    )

    assert galaxy_class1.draw_deflector().mass_properties["gamma_pl"] == 2.05


def test_galaxy_deflectors_foreground_color_gradient():
    red_galaxies = foreground_test_galaxy_table()
    blue_galaxies = foreground_test_galaxy_table()
    kwargs_deflector_cut = {}
    kwargs_mass2light = {}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    foreground_color_gradient = {
        "component_spectral_slopes": [2.0, -1.0],
        "reference_band": "i",
    }

    galaxy_class = GalaxyDeflectors(
        red_galaxies,
        blue_galaxy_list=blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=kwargs_mass2light,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type=None,
        foreground_color_gradient=foreground_color_gradient,
        foreground_component_weights=(0.4, 0.6),
    )
    assert "color_gradient" not in red_galaxies.colnames
    assert "color_gradient" not in blue_galaxies.colnames
    deflector = galaxy_class.draw_deflector()
    model_list, kwargs_i = deflector.light_model_lenstronomy(band="i")
    _, kwargs_g = deflector.light_model_lenstronomy(band="g")
    _, kwargs_y = deflector.light_model_lenstronomy(band="y")

    assert model_list == ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"]
    assert kwargs_i[0]["R_sersic"] < kwargs_i[1]["R_sersic"]
    assert component_flux_fraction(kwargs_i) == pytest.approx(0.4)
    assert component_flux_fraction(kwargs_y) > component_flux_fraction(kwargs_g)


def test_foreground_color_gradient_rejects_unsupported_light_type():
    with pytest.raises(ValueError, match="requires a single_sersic or double_sersic"):
        GalaxyDeflectors(
            foreground_test_galaxy_table(),
            kwargs_mass2light={},
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
            sky_area=Quantity(value=0.05, unit="deg2"),
            catalog_type=None,
            light_type="catalog_source",
            foreground_color_gradient={"component_spectral_slopes": [2.0, -1.0]},
        )


def foreground_test_galaxy_table():
    return Table(
        {
            "z": [0.2, 0.3],
            "stellar_mass": [10**11, 2 * 10**11],
            "angular_size": [0.7, 0.8],
            "ellipticity": [0.2, 0.25],
            "mag_i": [19.0, 20.0],
            "mag_g": [20.0, 21.0],
            "mag_y": [18.0, 19.0],
            "e1_light": [0.1, 0.1],
            "e2_light": [0.0, 0.0],
            "e1_mass": [0.1, 0.1],
            "e2_mass": [0.0, 0.0],
            "n_sersic": [4.0, 4.0],
            "vel_disp": [200.0, 210.0],
        }
    )


def component_flux_fraction(kwargs_light):
    flux0 = 10 ** (-kwargs_light[0]["magnitude"] / 2.5)
    flux1 = 10 ** (-kwargs_light[1]["magnitude"] / 2.5)
    return flux0 / (flux0 + flux1)


if __name__ == "__main__":
    pytest.main()
