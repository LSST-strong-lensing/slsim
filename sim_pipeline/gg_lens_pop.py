from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.gg_lens import GGLens
import numpy as np


class GGLensPop(object):
    """
    class to perform samples of galaxy-galaxy lensing
    """

    def __init__(self, lens_type='early-type', source_type='galaxies', kwargs_lens_cut=None, kwargs_source_cut=None,
                 kwargs_mass2light=None, skypy_config=None, f_sky=0.1, cosmo=None):
        """

        :param lens_type: type of the lens
        :type lens_type: string
        :param source_type: type of the source
        :type source_type: string
        :param kwargs_selection:
        :type kwargs_selection: dict
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string
        :param f_sky: sky area (in deg^2)
        :type f_sky: float
        """
        pipeline = SkyPyPipeline(skypy_config=skypy_config, f_sky=f_sky)
        if kwargs_lens_cut is None:
            kwargs_lens_cut = {}
        if kwargs_mass2light is None:
            kwargs_mass2light = {}
        if lens_type == 'early-type':
            from sim_pipeline.Lenses.early_type_lens_galaxies import EarlyTypeLensGalaxies
            self._lens_galaxies = EarlyTypeLensGalaxies(pipeline.red_galaxies, kwargs_cut=kwargs_lens_cut,
                                                        kwargs_mass2light=kwargs_mass2light, cosmo=cosmo)
        else:
            raise ValueError('lens_type %s is not supported' % lens_type)

        if kwargs_source_cut is None:
            kwargs_source_cut = {}
        if source_type == 'galaxies':
            from sim_pipeline.Sources.galaxies import Galaxies
            self._source_galaxies = Galaxies(pipeline.blue_galaxies, kwargs_cut=kwargs_source_cut, cosmo=cosmo)
        else:
            raise ValueError('source_type %s is not supported' % source_type)
        self.cosmo = cosmo
        self.f_sky = f_sky

    def select_lens_at_random(self):
        """
        draw a random lens within the cuts of the lens and source, with possible additional cut in the lensing
        configuration.

        #TODO: make sure mass function is preserved, as well as option to draw all lenses within the cuts within the area

        :return: GGLens() instance with parameters of the deflector and lens and source light
        """
        source = self._source_galaxies.draw_galaxy()
        lens = self._lens_galaxies.draw_deflector()
        gg_lens = GGLens(deflector_dict=lens, source_dict=source, cosmo=self.cosmo)
        return gg_lens

    def draw_population(self, testarea= 90*np.pi):
        """
        return full population list of all lenses within the area
        # TODO: need to implement a version of it. (improve the algorithm)
            Draw a population of galaxy-galaxy lenses within the area.

        :return: List of GGLens instances with parameters of the deflectors and lens and source light.
        :rtype: list
        """

        # Initialize an empty list to store the GGLens instances
        gg_lens_population = []
        # Estimate the number of lensing systems
        num_lenses = self._lens_galaxies.deflector_number()
        num_sources = self._source_galaxies.galaxies_number()
        num_sources_tested_mean = (testarea * self.f_sky * num_sources)/12960000
        print(num_sources_tested_mean)
        print(np.int(num_lenses*num_sources_tested_mean))

# Draw a population of galaxy-galaxy lenses within the area.
        for _ in range(num_lenses):
            lens = self._lens_galaxies.draw_deflector()
            for _ in range(max(1,int(np.rint(np.random.normal
                                                 (loc=num_sources_tested_mean, scale=num_sources_tested_mean/5))))):
                source = self._source_galaxies.draw_galaxy()
                gg_lens = GGLens(deflector_dict=lens, source_dict=source, cosmo=self.cosmo, test_area = testarea)
                # Check the validity of the lens system
                if gg_lens.validity_test():
                    gg_lens_population.append(gg_lens)
        return len(gg_lens_population)