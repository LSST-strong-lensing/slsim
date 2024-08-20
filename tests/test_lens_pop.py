import numpy as np
import pytest
import os
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity

from slsim.lens_pop import LensPop
from slsim.lens_pop import draw_test_area


def create_lens_pop_instance(return_kext=False):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")
    kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}
    return LensPop(
        sky_area=sky_area,
        cosmo=cosmo,
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
    )


@pytest.fixture
def gg_lens_pop_instance():
    # Create LensPop instance without return_kext
    return create_lens_pop_instance(return_kext=False)


def test_pes_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")
    kwargs_deflector_cut = {"z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 26, "z_min": 0.1, "z_max": 5.0}
    pes_lens_pop = LensPop(
        deflector_type="all-galaxies",
        source_type="quasar_plus_galaxies",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        variability_model="sinusoidal",
        kwargs_variability={"amp", "freq"},
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"


def test_galaxies_lens_pop_halo_model_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.001, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

    g_lens_halo_model_pop = LensPop(
        deflector_type="halo-models",
        source_type="galaxies",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        kwargs_mass2light=None,
        skypy_config=None,
        slhammocks_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
    )
    assert g_lens_halo_model_pop._lens_galaxies.draw_deflector()["halo_mass"] != 0


def test_cluster_catalog_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.001, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

    path = os.path.dirname(__file__)
    module_path = os.path.dirname(path)
    cluster_catalog = os.path.join(module_path, "data/redMaPPer/clusters_example.fits")
    members_catalog = os.path.join(module_path, "data/redMaPPer/members_example.fits")
    cluster_config = {
        "cluster_catalog": cluster_catalog,
        "members_catalog": members_catalog
    }

    cluster_lens_cat_pop = LensPop(
        deflector_type="cluster-catalog",
        source_type="galaxies",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        kwargs_mass2light=None,
        skypy_config=None,
        cluster_config=cluster_config,
        sky_area=sky_area,
        cosmo=cosmo,
    )
    deflector = cluster_lens_cat_pop._lens_galaxies.draw_deflector()
    assert deflector["halo_mass"] > 0


def test_supernovae_plus_galaxies_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")
    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}
    pes_lens_pop = LensPop(
        deflector_type="all-galaxies",
        source_type="supernovae_plus_galaxies",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability={"MJD", "ps_mag_r"},
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        catalog_type="supernovae_sample",
        cosmo=cosmo,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"


def test_supernovae_plus_galaxies_lens_pop_instance_2():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.3, unit="deg2")
    sky_area1 = Quantity(value=0.1, unit="deg2")
    sky_area2 = Quantity(value=0.2, unit="deg2")
    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}
    time_range = np.linspace(-20, 50, 500)
    pes_lens_pop = LensPop(
        deflector_type="all-galaxies",
        source_type="supernovae_plus_galaxies",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability={"supernovae_lightcurve", "i"},
        sn_type="Ia",
        sn_absolute_mag_band="bessellb",
        sn_absolute_zpsys="ab",
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        source_sky_area=sky_area2,
        deflector_sky_area=sky_area1,
        cosmo=cosmo,
        lightcurve_time=time_range,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"
    assert "x_off" in pes_lens_class.source.source_dict.colnames
    assert len(
        pes_lens_class.source.kwargs_variability_extracted["i"]["ps_mag_i"]
    ) == len(time_range)
    assert pes_lens_pop.source_sky_area != pes_lens_pop.deflector_sky_area


def test_supernovae_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=3, unit="deg2")
    sky_area2 = Quantity(value=1, unit="deg2")
    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}
    time_range = np.linspace(-20, 50, 500)
    pes_lens_pop = LensPop(
        deflector_type="elliptical",
        source_type="supernovae",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability={"supernovae_lightcurve", "r"},
        sn_type="Ia",
        sn_absolute_mag_band="bessellb",
        sn_absolute_zpsys="ab",
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
        lightcurve_time=time_range,
    )
    large_skyarea = Quantity(value=3, unit="deg2")
    pes_lens_pop2 = LensPop(
        deflector_type="elliptical",
        source_type="supernovae",
        kwargs_deflector_cut=kwargs_deflector_cut,
        kwargs_source_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability={"supernovae_lightcurve", "r"},
        sn_type="Ia",
        sn_absolute_mag_band="bessellb",
        sn_absolute_zpsys="ab",
        kwargs_mass2light=None,
        skypy_config=None,
        source_sky_area=sky_area2,
        deflector_sky_area=sky_area2,
        sky_area=large_skyarea,
        cosmo=cosmo,
        lightcurve_time=time_range,
    )
    kwargs_lens_cuts = {}
    # drawing population
    pes_lens_population = pes_lens_pop.draw_population(
        speed_factor=1, kwargs_lens_cuts=kwargs_lens_cuts
    )
    pes_lens_population_speed = pes_lens_pop.draw_population(
        speed_factor=10, kwargs_lens_cuts=kwargs_lens_cuts
    )
    pes_lens_population2 = pes_lens_pop2.draw_population(
        speed_factor=1, kwargs_lens_cuts=kwargs_lens_cuts
    )
    pes_lens_population2_speed = pes_lens_pop2.draw_population(
        speed_factor=100, kwargs_lens_cuts=kwargs_lens_cuts
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_source"
    assert "z" in pes_lens_class.source.source_dict.colnames
    assert len(pes_lens_class.source.source_dict) == 1
    assert abs(len(pes_lens_population) - len(pes_lens_population2)) <= 12
    assert abs(len(pes_lens_population_speed) - len(pes_lens_population2_speed)) <= 12


def test_num_lenses_and_sources(gg_lens_pop_instance):
    num_lenses = gg_lens_pop_instance.deflector_number
    num_sources = gg_lens_pop_instance.source_number

    assert 100 <= num_lenses <= 6600, "Expected num_lenses to be between 5800 and 6600,"
    f"but got {num_lenses}"
    assert (
        10000 <= num_sources <= 100000
    ), "Expected num_sources to be between 1090000 and"
    f"1110000, but got {num_sources}"
    # assert 1 == 0


def test_num_sources_tested_and_test_area(gg_lens_pop_instance):
    lens = gg_lens_pop_instance._lens_galaxies.draw_deflector()
    test_area = draw_test_area(deflector=lens)
    assert (
        0.01 < test_area < 100 * np.pi
    ), "Expected test_area to be between 0.1 and 100*pi,"
    f"but got {test_area}"
    num_sources_range = gg_lens_pop_instance.get_num_sources_tested(testarea=test_area)
    assert (
        0 <= num_sources_range <= 50
    ), "Expected num_sources_range to be between 0 and 50,"
    f"but got {num_sources_range}"


if __name__ == "__main__":
    pytest.main()
