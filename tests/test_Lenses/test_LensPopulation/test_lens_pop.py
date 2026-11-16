import os
import pytest
import slsim
import pickle

import numpy as np
from numpy.testing import assert_raises
import slsim.Pipelines as pipelines
from slsim.Util.data_path import data_path

# import slsim.Deflectors as deflectors
from slsim.Deflectors.DeflectorPopulation.galaxy_deflectors import GalaxyDeflectors
from slsim.Deflectors.DeflectorPopulation.compound_lens_halos_galaxies import (
    CompoundLensHalosGalaxies,
)
from slsim.Deflectors.DeflectorPopulation.cluster_deflectors import ClusterDeflectors
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Sources.SourceCatalogues.SupernovaeCatalog.supernovae_sample import (
    SupernovaeCatalog,
)
from slsim.Sources.SourcePopulation.point_plus_extended_sources import (
    PointPlusExtendedSources,
)
from slsim.Sources.SourceCatalogues.QuasarCatalog.simple_quasar import (
    quasar_catalog_simple,
)

from astropy.units import Quantity
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from slsim.Lenses.LensPopulation.lens_pop import LensPop
from slsim.Lenses.LensPopulation.lens_pop import area_theta_e_infinity
from slsim.Lenses.lens import Lens

sky_area = Quantity(value=0.05, unit="deg2")
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None,
    sky_area=sky_area,
    filters=None,
)

try:
    import jax

    print(jax.__path__)
    use_jax = True
except ImportError:
    use_jax = False


def create_lens_pop_instance(return_kext=False):

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.05, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

    lens_galaxies = GalaxyDeflectors(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    kwargs = {"extended_source_type": "single_sersic"}
    source_galaxies = Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        **kwargs,
    )

    lenspop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=use_jax,
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

    lens_galaxies = GalaxyDeflectors(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    quasar_galaxies = quasar_catalog_simple(**{})
    kwargs = {
        "kwargs_variability": None,
        "variability_model": "light_curve",
    }
    source_galaxies = PointPlusExtendedSources(
        point_plus_extended_sources_list=quasar_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut=kwargs_source_cut,
        point_source_type="quasar",
        extended_source_type=None,
        joint_point_source_kwargs=kwargs,
    )

    pes_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=use_jax,
    )

    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(verbose=True, **kwargs_lens_cut)
    assert isinstance(pes_lens_class, Lens)


def test_galaxies_lens_pop_halo_model_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.001, "z_max": 2.5}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}

    halo_galaxy_simulation_pipeline = pipelines.SLHammocksPipeline(
        slhammocks_config=None, sky_area=sky_area, cosmo=cosmo, z_min=0.01, z_max=5.0
    )

    lens_galaxies = CompoundLensHalosGalaxies(
        deflector_table=halo_galaxy_simulation_pipeline._pipeline,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    kwargs = {"extended_source_type": "single_source"}
    source_galaxies = Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        **kwargs,
    )

    g_lens_halo_model_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=use_jax,
    )
    assert (
        g_lens_halo_model_pop._lens_galaxies.draw_deflector().mass_properties[
            "halo_mass"
        ]
        != 0
    )


def test_cluster_lens_pop_instance():
    np.random.seed(41)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"z_min": 0.2, "z_max": 1.0}
    kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.25, "z_max": 5.0}

    cluster_catalog_path = data_path("redMaPPer", "clusters_example.fits")
    members_catalog_path = data_path("redMaPPer", "members_example.fits")
    cluster_catalog = Table.read(cluster_catalog_path)
    members_catalog = Table.read(members_catalog_path)

    galaxies = Galaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    lens_clusters = ClusterDeflectors(
        cluster_list=cluster_catalog,
        members_list=members_catalog,
        galaxies=galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    kwargs = {"extended_source_type": "single_sersic"}
    source_galaxies = Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        **kwargs,
    )

    cluster_lens_pop = LensPop(
        deflector_population=lens_clusters,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=use_jax,
    )

    kwargs_lens_cut = {}
    pes_lens_class = cluster_lens_pop.select_lens_at_random(
        test_area=4 * np.pi, **kwargs_lens_cut
    )
    assert pes_lens_class.deflector.deflector_type == "group"
    kwargs_model, kwargs_params = pes_lens_class.lenstronomy_kwargs(band="g")
    assert len(kwargs_model["lens_model_list"]) >= 3  # halo, 1>= subhalo, LoS
    assert len(kwargs_model["lens_light_model_list"]) >= 1  # 1>= member galaxy
    assert pes_lens_class.deflector_velocity_dispersion() > 100


def test_galaxies_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.001, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    lens_galaxies = GalaxyDeflectors(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area,
    )

    path = (
        os.path.dirname(slsim.__file__)
        + "/Sources/SourceCatalogues/SupernovaeCatalog/supernovae_data.pkl"
    )
    with open(path, "rb") as f:
        supernovae_data = pickle.load(f)
    kwargs = {"extended_source_type": "single_sersic"}
    source_galaxies = Galaxies(
        galaxy_list=supernovae_data,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut=kwargs_source_cut,
        **kwargs,
    )

    gg_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=use_jax,
    )
    kwargs_lens_cut = {"mag_arc_limit": {"i": 30}}
    pes_lens_class = gg_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert isinstance(pes_lens_class, Lens)

    # test that we can't find a lens with a lensed arc of magnitude 6
    with assert_raises(ValueError):
        kwargs_lens_cut = {"mag_arc_limit": {"i": 6}}
        gg_lens_pop.select_lens_at_random(**kwargs_lens_cut)


def test_supernovae_plus_galaxies_lens_pop_instance_2():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector_sky_area = Quantity(value=0.1, unit="deg2")
    source_sky_area = Quantity(value=0.2, unit="deg2")
    sky_area = Quantity(value=0.05, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    time_range = np.linspace(-20, 50, 500)

    lens_galaxies = GalaxyDeflectors(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        blue_galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=deflector_sky_area,
    )

    supernovae_catalog = SupernovaeCatalog(
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

    point_source_kwargs = {
        "variability_model": "light_curve",
        "kwargs_variability": ["supernovae_lightcurve", "i"],
        "sn_type": "Ia",
        "sn_absolute_mag_band": "bessellb",
        "sn_absolute_zpsys": "ab",
        "lightcurve_time": time_range,
        "sn_modeldir": None,
    }
    source_galaxies = PointPlusExtendedSources(
        point_plus_extended_sources_list=supernovae_data,
        cosmo=cosmo,
        sky_area=source_sky_area,
        kwargs_cut=kwargs_source_cut,
        point_source_type="supernova",
        extended_source_type="single_sersic",
        joint_point_source_kwargs=point_source_kwargs,
    )

    pes_lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=sky_area,
        use_jax=use_jax,
    )
    kwargs_lens_cut = {}
    pes_lens_class = pes_lens_pop.select_lens_at_random(**kwargs_lens_cut)
    assert pes_lens_class._source[0].source_type == "point_plus_extended"
    assert "x_off" in supernovae_data.colnames


def test_supernovae_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area_1 = Quantity(value=0.1, unit="deg2")
    sky_area_pop = Quantity(value=0.1, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    time_range = np.linspace(-20, 50, 500)

    lens_galaxies_1 = GalaxyDeflectors(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area_1,
    )

    supernovae_catalog_1 = SupernovaeCatalog(
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

    point_source_kwargs = {
        "variability_model": "light_curve",
        "kwargs_variability": ["supernovae_lightcurve", "i"],
        "sn_type": "Ia",
        "sn_absolute_mag_band": "bessellb",
        "sn_absolute_zpsys": "ab",
        "lightcurve_time": time_range,
        "sn_modeldir": None,
    }
    source_galaxies_1 = PointPlusExtendedSources(
        point_plus_extended_sources_list=supernovae_data_1,
        cosmo=cosmo,
        sky_area=sky_area_1,
        kwargs_cut=kwargs_source_cut,
        point_source_type="supernova",
        extended_source_type=None,
        joint_point_source_kwargs=point_source_kwargs,
    )

    ps_lens_pop_1 = LensPop(
        deflector_population=lens_galaxies_1,
        source_population=source_galaxies_1,
        cosmo=cosmo,
        sky_area=sky_area_pop,
        use_jax=use_jax,
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
    assert ps_lens_class._source[0].source_type == "point_source"
    assert "z" in supernovae_data_1.colnames
    assert abs(len(ps_lens_population_1) - len(ps_lens_population_1_speed)) <= 12
    with pytest.raises(ValueError):
        LensPop(
            deflector_population=lens_galaxies_1,
            source_population=source_galaxies_1,
            cosmo=cosmo,
            use_jax=use_jax,
        )


def test_kilonova_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area_1 = Quantity(value=0.1, unit="deg2")
    sky_area_pop = Quantity(value=0.1, unit="deg2")

    kwargs_deflector_cut = {"band": "g", "band_max": 23, "z_min": 0.01, "z_max": 2.5}
    kwargs_source_cut = {"z_min": 0.1, "z_max": 5.0}

    time_range = np.linspace(0.1, 30, 50)

    lens_galaxies_1 = GalaxyDeflectors(
        red_galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=sky_area_1,
    )

    kilonova_data_1 = Table(
        {
            "z": [0.8, 1.0, 1.2],
            "ra_off": [0.001, 0.001, 0.001],
            "dec_off": [0.005, 0.005, 0.005],
            "ps_mag_i": [28.3, 28.5, 28.7],
            "ps_mag_r": [28.0, 28.2, 28.4],
        }
    )

    kwargs_kilonova = {
        "ejecta_mass": [0.01, 0.02, 0.03],
        "ejecta_velocity": [0.1, 0.2, 0.3],
        "opacity": [0.5, 3.0, 10.0],
        "temperature_floor": [5000, 4000, 3000],
        "kappa_gamma": 10,
    }

    point_source_kwargs = {
        "variability_model": "light_curve",
        "kwargs_variability": ["kilonova_lightcurve", "i", "r"],
        "lightcurve_time": time_range,
        "model_name": "mosfit_kilonova",
        "mag_zpsys": "AB",
        "modeldir": None,
        "kwargs_kilonova": kwargs_kilonova,
    }
    source_galaxies_1 = PointPlusExtendedSources(
        point_plus_extended_sources_list=kilonova_data_1,
        cosmo=cosmo,
        sky_area=sky_area_1,
        kwargs_cut=kwargs_source_cut,
        point_source_type="kilonova",
        extended_source_type=None,
        joint_point_source_kwargs=point_source_kwargs,
    )

    ps_lens_pop_1 = LensPop(
        deflector_population=lens_galaxies_1,
        source_population=source_galaxies_1,
        cosmo=cosmo,
        sky_area=sky_area_pop,
        use_jax=use_jax,
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
    assert isinstance(ps_lens_class, Lens)
    assert ps_lens_class._source[0].source_type == "point_source"
    assert "z" in kilonova_data_1.colnames
    assert abs(len(ps_lens_population_1) - len(ps_lens_population_1_speed)) <= 12
    with pytest.raises(ValueError):
        LensPop(
            deflector_population=lens_galaxies_1,
            source_population=source_galaxies_1,
            cosmo=cosmo,
            use_jax=use_jax,
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
    test_area = area_theta_e_infinity(
        theta_e_infinity=lens.theta_e_infinity(cosmo=cosmo)
    )
    assert (
        0.001 < test_area < 1000 * np.pi
    ), "Expected test_area to be between 0.1 and 100*pi,"
    f"but got {test_area}"
    num_sources_range = gg_lens_pop_instance.get_num_sources_tested(testarea=test_area)
    assert (
        0 <= num_sources_range <= 50
    ), "Expected num_sources_range to be between 0 and 50,"
    f"but got {num_sources_range}"


def test_los_pop_distribution_cache():
    from slsim.LOS.los_pop import LOSPop

    los_pop = LOSPop(los_bool=True)
    dist1 = los_pop._get_los_distribution()
    dist2 = los_pop._get_los_distribution()
    assert dist1 is dist2
    draw = los_pop.draw_los(source_redshift=1.5, deflector_redshift=0.5)
    assert np.isfinite(draw.convergence)


def test_deflector_vectorized_theta_e_infinity():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    area = Quantity(value=0.05, unit="deg2")
    n = 6

    epl_table = Table(
        {
            "z": np.linspace(0.2, 0.7, n),
            "vel_disp": np.linspace(180.0, 280.0, n),
            "angular_size": np.full(n, 0.5),
            "e1": np.zeros(n),
            "e2": np.zeros(n),
            "mag_g": np.full(n, 19.0),
        }
    )
    epl_pop = GalaxyDeflectors(
        red_galaxy_list=epl_table,
        kwargs_cut={"band": "g", "band_max": 25, "z_min": 0.1, "z_max": 1.0},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=area,
        catalog_type=None,
    )
    vec_epl = epl_pop._theta_e_infinity_for_selected(np.arange(n), cosmo)
    ref_epl = np.array(
        [
            epl_pop._draw_deflector_at_selected(i).theta_e_infinity(
                cosmo, use_jax=False
            )
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(vec_epl, ref_epl, rtol=1e-6)

    comp_table = Table(
        {
            "z": np.linspace(0.25, 0.65, n),
            "halo_mass": np.full(n, 2e13),
            "concentration": np.full(n, 6.0),
            "stellar_mass": np.full(n, 2e11),
            "angular_size": np.full(n, 0.6),
            "e1": np.zeros(n),
            "e2": np.zeros(n),
            "mag_i": np.full(n, 19.0),
        }
    )
    comp_pop = CompoundLensHalosGalaxies(
        deflector_table=comp_table,
        kwargs_cut={"band": "i", "band_max": 25, "z_min": 0.1, "z_max": 1.0},
        cosmo=cosmo,
        sky_area=area,
        catalog_type=None,
    )
    vec_comp = comp_pop._theta_e_infinity_for_selected(np.arange(n), cosmo)
    ref_comp = np.array(
        [
            comp_pop._draw_deflector_at_selected(i).theta_e_infinity(
                cosmo, use_jax=False
            )
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(vec_comp, ref_comp, rtol=1e-2)


def test_draw_population_fallbacks_and_edge_cases(gg_lens_pop_instance, monkeypatch):
    assert gg_lens_pop_instance.draw_population({}, speed_factor=1e9) == []

    # Subclass overriding get_num_sources_tested triggers safe generic fallback.
    class CustomLensPop(LensPop):
        def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
            return super().get_num_sources_tested(
                testarea=testarea, num_sources_tested_mean=num_sources_tested_mean
            )

    custom_pop = CustomLensPop(
        deflector_population=gg_lens_pop_instance._lens_galaxies,
        source_population=gg_lens_pop_instance._sources,
        cosmo=gg_lens_pop_instance.cosmo,
        sky_area=gg_lens_pop_instance.sky_area,
        use_jax=use_jax,
    )
    assert custom_pop._try_draw_population_vectorized(1, {}, False, 10, False) is None
    assert isinstance(custom_pop.draw_population({}, speed_factor=500), list)

    # Non-finite ra_off falls back cleanly without failing vectorized screening.
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    area = Quantity(value=0.05, unit="deg2")
    ps_table = Table(
        {
            "z": [0.9, 1.2],
            "ra_off": [0.001, np.nan],
            "dec_off": [0.002, 0.003],
            "ps_mag_i": [22.0, 22.5],
        }
    )
    ps_src = PointPlusExtendedSources(
        point_plus_extended_sources_list=ps_table,
        cosmo=cosmo,
        sky_area=area,
        kwargs_cut={"z_min": 0.1, "z_max": 3.0},
        point_source_type="quasar",
        extended_source_type=None,
    )
    ps_pop = LensPop(
        deflector_population=gg_lens_pop_instance._lens_galaxies,
        source_population=ps_src,
        cosmo=cosmo,
        sky_area=area,
        use_jax=use_jax,
    )
    assert ps_pop._extract_source_screen_arrays() is None
    assert isinstance(ps_pop.draw_population({}, speed_factor=200), list)


def test_deflector_vectorized_all_branches():
    from astropy import units as u
    from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    area = Quantity(value=0.05, unit="deg2")
    n = 4
    base_kwargs = {
        "kwargs_cut": {"band": "g", "band_max": 25, "z_min": 0.05, "z_max": 1.0},
        "kwargs_mass2light": {},
        "cosmo": cosmo,
        "sky_area": area,
        "catalog_type": None,
    }

    # EPL via _vel_disp_from_stellar_mass and m_star_v_disp_scaling
    epl_mstar_tbl = Table(
        {
            "z": np.linspace(0.2, 0.6, n),
            "stellar_mass": np.full(n, 2e11),
            "angular_size": np.full(n, 0.5),
            "e1": np.zeros(n),
            "e2": np.zeros(n),
            "mag_g": np.full(n, 19.0),
        }
    )
    pop_interp = DeflectorsBase(
        deflector_table=epl_mstar_tbl.copy(), mass_type="EPL", **base_kwargs
    )
    pop_interp._vel_disp_from_stellar_mass = lambda log_m: np.full_like(
        log_m, 230.0, dtype=float
    )
    assert np.allclose(
        pop_interp._theta_e_infinity_for_selected(np.arange(n), cosmo),
        pop_interp._draw_deflector_at_selected(0).theta_e_infinity(
            cosmo, use_jax=False
        ),
        rtol=1e-6,
    )

    m2l_kwargs = dict(base_kwargs)
    m2l_kwargs["kwargs_mass2light"] = {"m_star_v_disp_scaling": True}
    pop_scaling = DeflectorsBase(
        deflector_table=epl_mstar_tbl.copy(), mass_type="EPL", **m2l_kwargs
    )
    assert np.allclose(
        pop_scaling._theta_e_infinity_for_selected(np.arange(n), cosmo),
        pop_scaling._draw_deflector_at_selected(0).theta_e_infinity(
            cosmo, use_jax=False
        ),
        rtol=1e-6,
    )

    # Fallback None cases for EPL
    pop_no_vdisp = DeflectorsBase(
        deflector_table=epl_mstar_tbl.copy(), mass_type="EPL", **base_kwargs
    )
    assert pop_no_vdisp._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    bad_mstar_tbl = epl_mstar_tbl.copy()
    bad_mstar_tbl["stellar_mass"][0] = -1.0
    pop_bad_mstar = DeflectorsBase(
        deflector_table=bad_mstar_tbl, mass_type="EPL", **m2l_kwargs
    )
    assert pop_bad_mstar._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    bad_vdisp_tbl = Table(
        {
            "z": np.linspace(0.2, 0.6, n),
            "vel_disp": np.array([200.0, -5.0, 210.0, 220.0]),
            "angular_size": np.full(n, 0.5),
            "e1": np.zeros(n),
            "e2": np.zeros(n),
            "mag_g": np.full(n, 19.0),
        }
    )
    pop_bad_vdisp = DeflectorsBase(
        deflector_table=bad_vdisp_tbl, mass_type="EPL", **base_kwargs
    )
    assert pop_bad_vdisp._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    # NFW-only and HERNQUIST-only (physical_size + Quantity + skypy scaling)
    nfw_tbl = Table(
        {
            "z": np.linspace(0.2, 0.6, n),
            "halo_mass": np.full(n, 2e14),
            "concentration": np.full(n, 5.0),
            "physical_size": np.full(n, 4.0) * u.kpc,
            "stellar_mass": np.full(n, 2e11),
            "e1": np.zeros(n),
            "e2": np.zeros(n),
            "mag_g": np.full(n, 19.0),
        }
    )
    pop_nfw = DeflectorsBase(
        deflector_table=nfw_tbl.copy(),
        mass_type="NFW",
        light_type="single_sersic",
        **base_kwargs,
    )
    vec_nfw = pop_nfw._theta_e_infinity_for_selected(np.arange(n), cosmo)
    ref_nfw = pop_nfw._draw_deflector_at_selected(0).theta_e_infinity(
        cosmo, use_jax=False
    )
    np.testing.assert_allclose(vec_nfw[0], ref_nfw, rtol=1e-2)

    hern_kwargs = dict(base_kwargs)
    hern_kwargs["catalog_type"] = "skypy"
    pop_hern = DeflectorsBase(
        deflector_table=nfw_tbl.copy(),
        mass_type="HERNQUIST",
        light_type="hernquist",
        **hern_kwargs,
    )
    assert np.all(pop_hern._theta_e_infinity_for_selected(np.arange(n), cosmo) > 0.0)

    # Guard edge cases returning None / empty
    assert len(pop_nfw._theta_e_infinity_for_selected([], cosmo)) == 0

    bad_z_tbl = nfw_tbl.copy()
    pop_bad_z = DeflectorsBase(
        deflector_table=bad_z_tbl,
        mass_type="NFW",
        kwargs_mass2light={},
        kwargs_cut={},
        cosmo=cosmo,
        sky_area=area,
        catalog_type=None,
    )
    pop_bad_z._objects_select["z"][1] = -0.1
    assert pop_bad_z._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    pop_pjaffe = DeflectorsBase(
        deflector_table=nfw_tbl.copy(), mass_type="PJAFFE", **base_kwargs
    )
    assert pop_pjaffe._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    list_defs = [
        Table(
            {
                "z": [0.3],
                "vel_disp": [220.0],
                "angular_size": [0.5],
                "e1": [0.0],
                "e2": [0.0],
                "mag_g": [19.0],
            }
        )
    ]
    pop_list = DeflectorsBase(deflector_table=list_defs, mass_type="EPL", **base_kwargs)
    assert pop_list._theta_e_infinity_for_selected([0], cosmo) is None

    no_halo_tbl = epl_mstar_tbl.copy()
    pop_comp_missing_nfw = DeflectorsBase(
        deflector_table=no_halo_tbl, mass_type="NFW_HERNQUIST", **base_kwargs
    )
    assert (
        pop_comp_missing_nfw._theta_e_infinity_for_selected(np.arange(n), cosmo) is None
    )

    no_size_tbl = nfw_tbl.copy()
    no_size_tbl.remove_column("physical_size")
    pop_comp_missing_size = DeflectorsBase(
        deflector_table=no_size_tbl, mass_type="NFW_HERNQUIST", **base_kwargs
    )
    assert (
        pop_comp_missing_size._theta_e_infinity_for_selected(np.arange(n), cosmo)
        is None
    )

    pop_bad_light = DeflectorsBase(
        deflector_table=nfw_tbl.copy(),
        mass_type="HERNQUIST",
        light_type="double_sersic",
        **base_kwargs,
    )
    assert pop_bad_light._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    pop_no_mstar = DeflectorsBase(
        deflector_table=bad_vdisp_tbl, mass_type="HERNQUIST", **base_kwargs
    )
    assert pop_no_mstar._theta_e_infinity_for_selected(np.arange(n), cosmo) is None

    pop_bernardi = Galaxies(
        galaxy_list=nfw_tbl.copy(),
        cosmo=cosmo,
        sky_area=area,
        kwargs_cut=base_kwargs["kwargs_cut"],
        catalog_type="skypy",
        size_model="Bernardi",
    )
    # Trigger _size_model == "Bernardi" and "skypy" on DeflectorsBase instance
    pop_hern._size_model = "Bernardi"
    assert pop_hern._theta_e_infinity_for_selected(np.arange(n), cosmo) is None
    assert pop_bernardi.source_number_selected == n


def test_draw_population_vectorized_sources_and_fallbacks():
    from slsim.Sources.SourcePopulation.point_sources import PointSources

    np.random.seed(42)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    def_area = Quantity(value=0.001, unit="deg2")
    src_area = Quantity(value=100.0, unit="arcsec2")
    pop_area = Quantity(value=0.001, unit="deg2")

    n_def = 4
    lens_tbl = Table(
        {
            "z": np.full(n_def, 0.25),
            "vel_disp": np.full(n_def, 320.0),
            "angular_size": np.full(n_def, 0.5),
            "e1": np.full(n_def, 0.05),
            "e2": np.zeros(n_def),
            "mag_g": np.full(n_def, 18.0),
            "mag_i": np.full(n_def, 17.5),
        }
    )
    def_pop = GalaxyDeflectors(
        red_galaxy_list=lens_tbl,
        kwargs_cut={"band": "g", "band_max": 25, "z_min": 0.05, "z_max": 1.0},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=def_area,
        catalog_type=None,
    )

    n_src = 8
    pes_tbl = Table(
        {
            "z": np.full(n_src, 1.5),
            "ps_mag_i": np.full(n_src, 20.0),
            "mag_i": np.full(n_src, 21.0),
            "angular_size": np.full(n_src, 0.15),
            "e1": np.zeros(n_src),
            "e2": np.zeros(n_src),
            "ra_off": np.full(n_src, 0.001),
            "dec_off": np.full(n_src, 0.002),
        }
    )
    qso_kw = {
        "variability_model": "light_curve",
        "kwargs_variability": None,
        "agn_driving_variability_model": None,
        "agn_driving_kwargs_variability": None,
        "lightcurve_time": None,
    }
    pes_src = PointPlusExtendedSources(
        point_plus_extended_sources_list=pes_tbl,
        cosmo=cosmo,
        sky_area=src_area,
        kwargs_cut={"z_min": 0.1, "z_max": 3.0},
        point_source_type="quasar",
        extended_source_type="single_sersic",
        joint_point_source_kwargs=qso_kw,
    )
    pes_pop = LensPop(
        deflector_population=def_pop,
        source_population=pes_src,
        cosmo=cosmo,
        sky_area=pop_area,
        use_jax=use_jax,
    )
    pes_lenses = pes_pop.draw_population({}, multi_source=True)
    assert len(pes_lenses) > 0

    # PointPlusExtendedSources without offset columns (joint defaults and zero-offset)
    pes_no_off_tbl = pes_tbl.copy()
    pes_no_off_tbl.remove_columns(["ra_off", "dec_off"])
    pes_src_zero_off = PointPlusExtendedSources(
        point_plus_extended_sources_list=pes_no_off_tbl.copy(),
        cosmo=cosmo,
        sky_area=src_area,
        kwargs_cut={"z_min": 0.1, "z_max": 3.0},
        point_source_type="quasar",
        extended_source_type=None,
        joint_point_source_kwargs=qso_kw,
    )
    assert (
        LensPop(
            def_pop, pes_src_zero_off, cosmo, sky_area=pop_area
        )._extract_source_screen_arrays()[1]
        is None
    )
    pes_src_joint_off = PointPlusExtendedSources(
        point_plus_extended_sources_list=pes_no_off_tbl.copy(),
        cosmo=cosmo,
        sky_area=src_area,
        kwargs_cut={"z_min": 0.1, "z_max": 3.0},
        point_source_type="quasar",
        extended_source_type=None,
        joint_point_source_kwargs={**qso_kw, "ra_off": 0.002, "dec_off": 0.001},
    )
    _, joint_ra, joint_dec = LensPop(
        def_pop, pes_src_joint_off, cosmo, sky_area=pop_area
    )._extract_source_screen_arrays()
    assert np.allclose(joint_ra, 0.002) and np.allclose(joint_dec, 0.001)

    # PointSources-only vectorized draw_population
    ps_only_tbl = Table({"z": np.full(n_src, 1.5), "ps_mag_i": np.full(n_src, 20.0)})
    ps_only_src = PointSources(
        point_source_list=ps_only_tbl,
        cosmo=cosmo,
        sky_area=src_area,
        kwargs_cut={"z_min": 0.1, "z_max": 3.0},
        point_source_type="quasar",
        joint_point_source_kwargs=qso_kw,
    )
    ps_only_lenses = LensPop(
        def_pop, ps_only_src, cosmo, sky_area=pop_area, use_jax=use_jax
    ).draw_population({})
    assert len(ps_only_lenses) > 0

    # Cover empty z_pass (z_d > all z_s) and non-linear bound (non-SIS EPL)
    from slsim.Util.cosmo_util import get_cosmo_cache

    _, _, _, empty_keep = LensPop._screen_candidates_for_deflector(
        z_d=3.0,
        theta_e_inf=2.0,
        n_test=5,
        src_z=np.array([1.0, 1.5]),
        src_ra_off=None,
        src_dec_off=None,
        cosmo_cache=get_cosmo_cache(cosmo),
        use_linear_bound=False,
        dds_ds_100=0.5,
    )
    assert empty_keep.size == 0

    non_sis_def = GalaxyDeflectors(
        red_galaxy_list=lens_tbl.copy(),
        kwargs_cut={"band": "g", "band_max": 25, "z_min": 0.05, "z_max": 1.0},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=def_area,
        catalog_type=None,
        gamma_pl=2.1,
        sis_convention=False,
    )
    assert isinstance(
        LensPop(non_sis_def, ps_only_src, cosmo, sky_area=pop_area).draw_population({}),
        list,
    )

    # Exercise remaining _try_draw_population_vectorized guards + generic unscreened loop
    class CustomMeanPop(LensPop):
        def get_num_sources_tested_mean(self, testarea):
            return super().get_num_sources_tested_mean(testarea)

    assert (
        CustomMeanPop(
            def_pop, ps_only_src, cosmo, sky_area=pop_area
        )._try_draw_population_vectorized(1, {}, False, 1, False)
        is None
    )

    # Generic loop succeeding with multi_source=False and multi_source=True
    generic_lenses = CustomMeanPop(
        def_pop, ps_only_src, cosmo, sky_area=pop_area
    ).draw_population({}, multi_source=False)
    assert len(generic_lenses) > 0
    assert (
        CustomMeanPop(
            def_pop, ps_only_src, cosmo, sky_area=pop_area
        )._draw_sources_unscreened(
            deflector=def_pop.draw_deflector(),
            test_area=10.0,
            num_sources_tested=0,
            kwargs_lens_cuts={},
            multi_source=True,
        )
        is None
    )
    assert (
        CustomMeanPop(
            def_pop, ps_only_src, cosmo, sky_area=pop_area
        )._draw_sources_unscreened(
            deflector=def_pop.draw_deflector(),
            test_area=1e6,
            num_sources_tested=1,
            kwargs_lens_cuts={"min_image_separation": 50.0},
            multi_source=True,
        )
        is None
    )

    class CustomDefPop(GalaxyDeflectors):
        def draw_deflector(self, z_max=None, z_min=None, deflector_index=None):
            return super().draw_deflector(
                z_max=z_max, z_min=z_min, deflector_index=deflector_index
            )

    custom_def = CustomDefPop(
        red_galaxy_list=lens_tbl.copy(),
        kwargs_cut={"band": "g", "band_max": 25, "z_min": 0.05, "z_max": 1.0},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=def_area,
        catalog_type=None,
    )
    assert (
        LensPop(
            custom_def, ps_only_src, cosmo, sky_area=pop_area
        )._try_draw_population_vectorized(1, {}, False, 1, False)
        is None
    )

    empty_src = PointSources(
        point_source_list=Table({"z": [1.5], "ps_mag_i": [20.0]}),
        cosmo=cosmo,
        sky_area=src_area,
        kwargs_cut={},
        point_source_type="quasar",
        joint_point_source_kwargs=qso_kw,
    )
    empty_src._num_select = 0
    assert (
        LensPop(
            def_pop, empty_src, cosmo, sky_area=pop_area
        )._try_draw_population_vectorized(1, {}, False, 1, False)
        is None
    )

    list_src = Galaxies(
        galaxy_list=[
            Table(
                {
                    "z": [1.5],
                    "angular_size": [0.2],
                    "e1": [0.0],
                    "e2": [0.0],
                    "mag_i": [21.0],
                }
            )
        ],
        cosmo=cosmo,
        sky_area=src_area,
        catalog_type=None,
    )
    pop_list_src = LensPop(def_pop, list_src, cosmo, sky_area=pop_area)
    assert not pop_list_src._is_source_population_vectorizable()
    assert pop_list_src._try_draw_population_vectorized(1, {}, False, 1, False) is None

    class CustomObjSrc(Galaxies):
        def draw_object(self, z_max=None, z_min=None, galaxy_index=None):
            return super().draw_object(
                z_max=z_max, z_min=z_min, galaxy_index=galaxy_index
            )

    class CustomDrawSrc(Galaxies):
        def draw_source(self, z_max=None, z_min=None, galaxy_index=None):
            return super().draw_source(
                z_max=z_max, z_min=z_min, galaxy_index=galaxy_index
            )

    src_tbl = Table(
        {
            "z": [1.5],
            "angular_size": [0.2],
            "e1": [0.0],
            "e2": [0.0],
            "mag_i": [21.0],
        }
    )
    assert not LensPop(
        def_pop,
        CustomObjSrc(
            galaxy_list=src_tbl.copy(),
            cosmo=cosmo,
            sky_area=src_area,
            catalog_type=None,
        ),
        cosmo,
        sky_area=pop_area,
    )._is_source_population_vectorizable()
    assert not LensPop(
        def_pop,
        CustomDrawSrc(
            galaxy_list=src_tbl.copy(),
            cosmo=cosmo,
            sky_area=src_area,
            catalog_type=None,
        ),
        cosmo,
        sky_area=pop_area,
    )._is_source_population_vectorizable()

    bad_src_z = PointSources(
        point_source_list=Table({"z": [1.5], "ps_mag_i": [20.0]}),
        cosmo=cosmo,
        sky_area=src_area,
        kwargs_cut={},
        point_source_type="quasar",
        joint_point_source_kwargs=qso_kw,
    )
    bad_src_z._objects_select["z"][0] = np.nan
    assert (
        LensPop(
            def_pop, bad_src_z, cosmo, sky_area=pop_area
        )._extract_source_screen_arrays()
        is None
    )

    no_vdisp_def = GalaxyDeflectors(
        red_galaxy_list=Table(
            {
                "z": [0.3],
                "vel_disp": [200.0],
                "angular_size": [0.5],
                "e1": [0.0],
                "e2": [0.0],
                "mag_g": [18.0],
            }
        ),
        kwargs_cut={},
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=def_area,
        catalog_type=None,
    )
    no_vdisp_def._objects_select["vel_disp"][0] = np.nan
    assert (
        LensPop(
            no_vdisp_def, ps_only_src, cosmo, sky_area=pop_area
        )._try_draw_population_vectorized(1, {}, False, 1, False)
        is None
    )


if __name__ == "__main__":
    pytest.main()
