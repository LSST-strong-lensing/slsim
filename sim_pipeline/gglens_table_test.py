from astropy.table import Table
from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.gg_lens import GGLens, theta_e_when_source_infinity
import numpy as np

def create_table(self, gg_lens_population):
         """
          Creates an astropy table for a given gg_lens population.
    
         :param gg_lens_population: List of GGLens instances.
         :type gg_lens_population: list
         :return: Astropy table with lens properties.
         :rtype: `~astropy.table.Table`
         """
    
            # Assuming the GGLens class has a method `get_einstein_radius` to get the Einstein radius
            # Adjust the following extraction based on the actual attributes/methods of GGLens
    
         einstein_radii = [lens.get_einstein_radius() for lens in gg_lens_population]

        # Select/Add/Removed desired parameters.
         t = Table([einstein_radii],names=('Einstein Radius'))
         return t


gg_lens_population_obj  = GGLens
kwargs_lens_cut = {}
#drawing the gg lens population
pop = gg_lens_population_obj.draw_population(kwargs_lens_cut)

#create the Astropy table
table = gg_lens_population_obj.create_table(pop)
# table.write('output_file.csv', format='csv')
print(table)


