from astropy.table import Table
from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.gg_lens import GGLens, theta_e_when_source_infinity

import numpy as np

def create_table(gg_lens_population):
    """
    Creates an astropy table for a given gg_lens population.

    :param gg_lens_population: List of GGLens instances.
    :type gg_lens_population: list
    :return: Astropy table with lens properties.
    :rtype: `~astropy.table.Table`
    """

    # get the various properties
    einstein_radii = [lens.get_einstein_radius() for lens in gg_lens_population]
    deflector_ellipticity = [lens.get_ellipticity() for lens in gg_lens_population]
    deflector_velocity_disperson = [lens.get_velocity_dispersion() for lens in gg_lens_population]
    deflector_stellar_mass = [lens.get_stellar_mass() for lens in gg_lens_population]
    deflector_magnitude = [lens.get_magnitude() for lens in gg_lens_population]
    source_redshift = [lens.get_source_redshift() for lens in gg_lens_population]
    lens_redshift = [lens.get_lens_redshift() for lens in gg_lens_population]

    t = Table([einstein_radii, deflector_ellipticity, deflector_velocity_disperson, deflector_stellar_mass, deflector_magnitude, source_redshift, lens_redshift],
              names=('Einstein Radius', 'Deflector Ellipticity', 'Deflector Velocity Dispersion', 'Deflector Stellar Mass', 'Deflector Magnitude', 'Source Redshift', 'Lens Redshift'))

    return t

gg_lens_population_obj = GGLens
kwargs_lens_cut = {}
# Drawing the gg lens population
pop = gg_lens_population_obj.draw_population(kwargs_lens_cut)

# Create the Astropy table
table = create_table(pop)
# table.write('output_file.csv', format='csv')
print(table)
