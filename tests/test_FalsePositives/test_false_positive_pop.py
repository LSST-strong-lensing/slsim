import os
import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy.table import Table

import slsim.Sources as sources
import slsim.Deflectors as deflectors
import slsim.Pipelines as pipelines
from slsim.Sources.SourcePopulation.point_sources import PointSources
from slsim.FalsePositives.false_positive_pop import FalsePositivePop

# --- Setup Mock Data and Pipelines ---
sky_area = Quantity(value=0.01, unit="deg2")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None, sky_area=sky_area
)

lens_galaxies = deflectors.EllipticalLensGalaxies(
    galaxy_list=galaxy_simulation_pipeline.red_galaxies,
    kwargs_cut={"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5},
    kwargs_mass2light=0.1,
    cosmo=cosmo,
    sky_area=sky_area,
)

source_galaxies = sources.Galaxies(
    galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
    kwargs_cut={"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0},
    cosmo=cosmo,
    sky_area=sky_area,
    catalog_type="skypy",
    extended_source_type="single_sersic",
)

path = os.path.dirname(__file__)
loaded_qso_host_catalog = Table.read(
    os.path.join(path, "../TestData/qso_host_catalog.fits")
)
source_quasars_high_z = PointSources(
    point_source_list=loaded_qso_host_catalog,
    kwargs_cut={"band": "g", "band_max": 28, "z_min": 2, "z_max": 5.0},
    cosmo=cosmo,
    sky_area=Quantity(12.0, unit="deg2"),
    point_source_type="quasar",
    point_source_kwargs={},
)


def test_false_positive_core_and_area():
    """Tests single population, area clustering, and base methods."""
    fp_pop = FalsePositivePop(
        central_galaxy_population=lens_galaxies,
        intruder_populations=source_galaxies,
        intruder_number_choices=[1],
        cosmo=cosmo,
    )

    # Check deflector logic explicitly
    draw_deflector, z_max = fp_pop.draw_deflector()
    assert z_max == draw_deflector.redshift + 0.002

    # Check source area drawing explicitly
    draw_source = fp_pop.draw_sources(z_max=z_max, test_area=0.1, theta_e=1.0)
    assert not isinstance(draw_source, list)  # Expecting single source instance

    # Check overall FP generation
    draw_fp = fp_pop.draw_false_positive(number=1)
    assert draw_fp.source_number == 1


def test_false_positive_multiple_and_ring():
    """Tests list of populations, list choices, weights, multiple items, and
    ring clustering."""
    fp_pop = FalsePositivePop(
        central_galaxy_population=lens_galaxies,
        intruder_populations=[source_galaxies],
        intruder_number_choices=[[2, 3]],
        weights_for_intruder_number=[[1.0, 0.0]],  # Guarantee a pick of 2
        cosmo=cosmo,
        clustering_mode="ring",
    )

    draw_fp_list = fp_pop.draw_false_positive(number=2)
    assert isinstance(draw_fp_list, list)
    assert len(draw_fp_list) == 2

    fp = draw_fp_list[0]
    assert fp.source_number == 2

    # Verify ring coordinates fall within [0.5, 2.5] * theta_e
    for i in range(fp.source_number):
        center_source = fp.source(i)._source._center_source
        r = np.sqrt(center_source[0] ** 2 + center_source[1] ** 2)
        assert (
            0.5 * fp.einstein_radius_infinity <= r <= 2.5 * fp.einstein_radius_infinity
        )


def test_false_positive_edge_cases():
    """Tests initialization branches, ValueErrors, empty sources, and retry
    loops."""
    # 1. Validation error branch
    with pytest.raises(ValueError):
        FalsePositivePop(
            central_galaxy_population=lens_galaxies,
            intruder_populations=[source_galaxies],
            intruder_number_choices=[[1], [2]],
        )

    # 2. List of populations but no weights branch
    fp_no_weights = FalsePositivePop(
        central_galaxy_population=lens_galaxies,
        intruder_populations=[source_galaxies, source_quasars_high_z],
        intruder_number_choices=[[1], [1]],
        cosmo=cosmo,
    )
    draw_fp_no_weights = fp_no_weights.draw_false_positive(number=1)
    assert draw_fp_no_weights is not None
    assert draw_fp_no_weights.source_number == 2

    # 3. Integer choice, weights provided for single pop, and 'not all_sources' empty branch
    fp_pop_int = FalsePositivePop(
        central_galaxy_population=lens_galaxies,
        intruder_populations=source_galaxies,
        intruder_number_choices=0,  # Forces n_draw = 0
        weights_for_intruder_number=[1.0],
        cosmo=cosmo,
    )
    assert fp_pop_int.draw_sources(z_max=5.0, test_area=0.1) is None

    # 4. High-z retry loop branch (triggers 'source is None')
    fp_pop_high_z = FalsePositivePop(
        central_galaxy_population=lens_galaxies,
        intruder_populations=source_quasars_high_z,
        intruder_number_choices=[1],
        cosmo=cosmo,
    )
    draw_fp_high_z = fp_pop_high_z.draw_false_positive()
    assert draw_fp_high_z is not None
    assert draw_fp_high_z.source_number == 1


if __name__ == "__main__":
    pytest.main()
