import pytest
from astropy.table import Table
from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.gg_lens import GGLens, theta_e_when_source_infinity
import numpy as np


def create_table(gg_lens_population):
    """Creates an astropy table for a given gg_lens population.

    :param gg_lens_population: List of GGLens instances.
    :type gg_lens_population: list
    :return: Astropy table with lens properties.
    :rtype: `~astropy.table.Table`
    """

    # Assuming the GGLens class has a method `get_einstein_radius` to get the Einstein radius
    einstein_radii = [lens.get_einstein_radius() for lens in gg_lens_population]

    # Create table with desired parameters.
    t = Table([einstein_radii], names=("Einstein_Radius"))
    return t


@pytest.fixture
def sample_gg_lens_population():
    gg_lens_population_obj = GGLens
    kwargs_lens_cut = {}
    return gg_lens_population_obj.draw_population(kwargs_lens_cut)


def test_create_table(sample_gg_lens_population):
    table = create_table(sample_gg_lens_population)
    assert isinstance(table, Table)
    assert "Einstein_Radius" in table.colnames
    # Add more assertions based on your expectations for the table
