import os
import pytest
import slsim
import pickle

import numpy as np
import slsim.Sources as sources
import slsim.Pipelines as pipelines
import slsim.Deflectors as deflectors

from astropy.units import Quantity
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from slsim.lens_pop import LensPop
from slsim.lens_pop import draw_test_area

sky_area = Quantity(value=0.05, unit="deg2")
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None,
    sky_area=sky_area,
    filters=None,
)


def create_lens_pop_instance(return_kext=False):

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

    lens_galaxies = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
    )

    lenspop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    return lenspop


@pytest.fixture
def gg_lens_pop_instance():
    # Create LensPop instance without return_kext
    return create_lens_pop_instance(return_kext=False)


def test_draw_population(gg_lens_pop_instance):
    lens_pop = gg_lens_pop_instance
    kwargs_lens_cuts = {}
    lens_population = lens_pop.draw_population(kwargs_lens_cuts, multi_source=True)
    lens_population2 = lens_pop.draw_population(kwargs_lens_cuts, multi_source=False)
    assert len(lens_population) <= 40
    assert len(lens_population2) <= 40


def test_pes_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 26, "z_min": 0.1, "z_max": 5.0}

    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    quasar_galaxies = sources.QuasarCatalog.quasar_galaxies_simple(**{})
    source_galaxies = sources.PointPlusExtendedSources(
        point_plus_extended_sources_list=quasar_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut=kwargs_source_cut,
        variability_model="sinusoidal",
        kwargs_variability_model={"amp", "freq"},
    )

    pes_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"


def test_galaxies_lens_pop_halo_model_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.001, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

    halo_galaxy_simulation_pipeline = pipelines.SLHammocksPipeline(
        slhammocks_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
    )

    lens_galaxies = deflectors.CompoundLensHalosGalaxies(
        halo_galaxy_list=halo_galaxy_simulation_pipeline._pipeline,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
    )

    g_lens_halo_model_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    assert g_lens_halo_model_pop._lens_galaxies.draw_deflector().halo_properties[0] != 0


def test_cluster_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.2, "z_max": 1.0}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.25, "z_max": 5.0}

    path = os.path.dirname(__file__)
    module_path = os.path.dirname(path)
    cluster_catalog_path = os.path.join(
        module_path, "data/redMaPPer/clusters_example.fits"
    )
    members_catalog_path = os.path.join(
        module_path, "data/redMaPPer/members_example.fits"
    )
    cluster_catalog = Table.read(cluster_catalog_path)
    members_catalog = Table.read(members_catalog_path)

    lens_clusters = deflectors.ClusterDeflectors(
        cluster_list=cluster_catalog,
        members_list=members_catalog,
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
    )

    cluster_lens_pop = LensPop(
        deflector_population=lens_clusters,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    kwargs_lens_cut = {}
    pes_lens_class = cluster_lens_pop.select_lens_at_random(
        test_area=4 * np.pi, **kwargs_lens_cut
    )
    assert pes_lens_class.deflector.deflector_type == "NFW_CLUSTER"
    kwargs_model, kwargs_params = pes_lens_class.lenstronomy_kwargs(band="g")
    assert len(kwargs_model["lens_model_list"]) >= 3  # halo, 1>= subhalo, LoS
    assert len(kwargs_model["lens_light_model_list"]) >= 1  # 1>= member galaxy
    assert pes_lens_class.deflector_velocity_dispersion() > 300


def test_supernovae_plus_galaxies_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    path = (
        os.path.dirname(slsim.__file__)
        + "/Sources/SupernovaeCatalog/supernovae_data.pkl"
    )
    with open(path, "rb") as f:
        supernovae_data = pickle.load(f)

    source_galaxies = sources.PointPlusExtendedSources(
        point_plus_extended_sources_list=supernovae_data,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability_model={"MJD", "ps_mag_r"},
        list_type="list",
    )

    pes_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"


def test_supernovae_plus_galaxies_lens_pop_instance_2():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector_sky_area = Quantity(value=0.1, unit="deg2")
    source_sky_area = Quantity(value=0.2, unit="deg2")
    sky_area = Quantity(value=0.05, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    time_range = np.linspace(-20, 50, 500)

    lens_galaxies = deflectors.AllLensGalaxies(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=deflector_sky_area,
    )

    supernovae_catalog = sources.SupernovaeCatalog.SupernovaeCatalog(
        sn_type="Ia",
        band_list=["i"],
        lightcurve_time=time_range,
        absolute_mag_band="bessellb",
        absolute_mag=None,
        mag_zpsys="ab",
        cosmo=cosmo,
        skypy_config=None,
        sky_area=source_sky_area,
    )
    supernovae_data = supernovae_catalog.supernovae_catalog(
        host_galaxy=True, lightcurve=False
    )

    source_galaxies = sources.PointPlusExtendedSources(
        point_plus_extended_sources_list=supernovae_data,
        cosmo=cosmo,
        sky_area=source_sky_area,
        kwargs_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability_model={"supernovae_lightcurve", "i"},
        lightcurve_time=time_range,
        sn_type="Ia",
        sn_absolute_mag_band="bessellb",
        sn_absolute_zpsys="ab",
        sn_modeldir=None,
    )

    pes_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source_type == "point_plus_extended"
    assert "x_off" in pes_lens_class.source[0].source_dict.colnames
    assert len(
        pes_lens_class.source[0].kwargs_variability_extracted["i"]["ps_mag_i"]
    ) == len(time_range)


def test_supernovae_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area_1 = Quantity(value=0.1, unit="deg2")
    sky_area_pop = Quantity(value=0.1, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    time_range = np.linspace(-20, 50, 500)

    lens_galaxies_1 = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area_1,
    )

    supernovae_catalog_1 = sources.SupernovaeCatalog.SupernovaeCatalog(
        sn_type="Ia",
        band_list=["r"],
        lightcurve_time=time_range,
        absolute_mag_band="bessellb",
        absolute_mag=None,
        mag_zpsys="ab",
        cosmo=cosmo,
        skypy_config=None,
        sky_area=sky_area_1,
    )
    supernovae_data_1 = supernovae_catalog_1.supernovae_catalog(
        host_galaxy=False, lightcurve=False
    )

    source_galaxies_1 = sources.PointSources(
        point_source_list=supernovae_data_1,
        cosmo=cosmo,
        sky_area=sky_area_1,
        kwargs_cut=kwargs_source_cut,
        variability_model="light_curve",
        kwargs_variability_model={"supernovae_lightcurve", "r"},
        lightcurve_time=time_range,
        sn_type="Ia",
        sn_absolute_mag_band="bessellb",
        sn_absolute_zpsys="ab",
    )

    ps_lens_pop_1 = LensPop(
        deflector_population=lens_galaxies_1,
        source_population=source_galaxies_1,
        cosmo=cosmo,
        sky_area=sky_area_pop,
    )
    # drawing population
    kwargs_lens_cuts = {}
    ps_lens_population_1 = ps_lens_pop_1.draw_population(
        speed_factor=1, kwargs_lens_cuts=kwargs_lens_cuts
    )
    ps_lens_population_1_speed = ps_lens_pop_1.draw_population(
        speed_factor=10, kwargs_lens_cuts=kwargs_lens_cuts
    )
    kwargs_lens_cut = {}
    ps_lens_class = ps_lens_pop_1.select_lens_at_random(**kwargs_lens_cut)
    assert ps_lens_class._source_type == "point_source"
    assert "z" in ps_lens_class.source[0].source_dict.colnames
    assert len(ps_lens_class.source[0].source_dict) == 1
    assert abs(len(ps_lens_population_1) - len(ps_lens_population_1_speed)) <= 12
    with pytest.raises(ValueError):
        LensPop(
            deflector_population=lens_galaxies_1,
            source_population=source_galaxies_1,
            cosmo=cosmo,
        )


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
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lens = gg_lens_pop_instance._lens_galaxies.draw_deflector()
    test_area = draw_test_area(v_sigma=lens.velocity_dispersion(cosmo=cosmo))
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
