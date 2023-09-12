from astropy.table import Table
from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.gg_lens import GGLens, theta_e_when_source_infinity
import numpy as np



def create_table(gg_lens_population, parameters=['einstein_radius', 'ellipticity', 'velocity_dispersion', 'stellar_mass', 'magnitude', 'source_redshift', 'lens_redshift']):
    """
    Creates an astropy table for a given gg_lens population.

    :param gg_lens_population: List of GGLens instances.
    :type gg_lens_population: list
    :param parameters: List of parameters to extract from the population.
    :type parameters: list
    :return: Astropy table with lens properties.
    :rtype: `~astropy.table.Table`
    """

    # Dictionary of methods to call for each parameter
    param_methods = {
        'einstein_radius': 'get_einstein_radius',
        'ellipticity': 'get_ellipticity',
        'velocity_dispersion': 'get_velocity_dispersion',
        'stellar_mass': 'get_stellar_mass',
        'magnitude': 'get_magnitude',
        'source_redshift': 'get_source_redshift',
        'lens_redshift': 'get_lens_redshift'
    }

    data = {param: [] for param in parameters}

    # Single loop through the gg_lens_population
    for lens in gg_lens_population:
        for param in parameters:
            method = getattr(lens, param_methods[param])
            data[param].append(method())

    # Construct table from the extracted data
    t = Table(names=parameters)
    for param in parameters:
        t[param] = data[param]

    return t

gg_lens_population_obj = GGLens
kwargs_lens_cut = {}
# Drawing the gg lens population
pop = gg_lens_population_obj.draw_population(kwargs_lens_cut)

# Create the Astropy table with desired parameters
desired_parameters = ['einstein_radius', 'ellipticity', 'source_redshift']
table = create_table(pop, parameters=desired_parameters)
# table.write('output_file.csv', format='csv')
print(table)
