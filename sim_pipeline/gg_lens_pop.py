from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.gg_lens import GGLens, theta_e_when_source_infinity
import numpy as np


def draw_test_area(deflector):
    """
    draw a test area around the deflector

    :param deflector: deflector dictionary
    :return: test area in arcsec^2
    """
    theta_e_infinity = theta_e_when_source_infinity(deflector)
    test_area = np.pi * (theta_e_infinity * 1.3) ** 2
    return test_area


class GGLensPop(object):
    """
    class to perform samples of galaxy-galaxy lensing
    """

    def __init__(self, lens_type='early-type', source_type='galaxies', kwargs_deflector_cut=None,
                 kwargs_source_cut=None, kwargs_mass2light=None, skypy_config=None, sky_area=None, filters=None,
                 cosmo=None):
        """

        :param lens_type: type of the lens
        :type lens_type: string
        :param source_type: type of the source
        :type source_type: string
        :param kwargs_deflector_cut: cuts on the deflector to be excluded in the sample
        :type kwargs_deflector_cut: dict
        :param kwargs_source_cut: cuts on the source to be excluded in the sample
        :type kwargs_source_cut: dict
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param filters: filters for SED integration
        :type filters: list of strings or None
        """
        pipeline = SkyPyPipeline(skypy_config=skypy_config, sky_area=sky_area, filters=filters)
        if kwargs_deflector_cut is None:
            kwargs_deflector_cut = {}
        if kwargs_mass2light is None:
            kwargs_mass2light = {}
        if lens_type == 'early-type':
            from sim_pipeline.Lenses.early_type_lens_galaxies import EarlyTypeLensGalaxies
            self._lens_galaxies = EarlyTypeLensGalaxies(pipeline.red_galaxies, kwargs_cut=kwargs_deflector_cut,
                                                        kwargs_mass2light=kwargs_mass2light, cosmo=cosmo,
                                                        sky_area=sky_area)
        elif lens_type == 'all-galaxies':
            from sim_pipeline.Lenses.all_lens_galaxies import AllLensGalaxies
            self._lens_galaxies = AllLensGalaxies(pipeline.red_galaxies, pipeline.blue_galaxies,
                                                  kwargs_cut=kwargs_deflector_cut, kwargs_mass2light=kwargs_mass2light,
                                                  cosmo=cosmo, sky_area=sky_area)
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
        self.f_sky = sky_area

    def select_lens_at_random(self, **kwargs_lens_cut):
        """
        draw a random lens within the cuts of the lens and source, with possible additional cut in the lensing
        configuration.

        #TODO: make sure mass function is preserved, as well as option to draw all lenses within the cuts within the area

        :return: GGLens() instance with parameters of the deflector and lens and source light
        """
        while True:
            source = self._source_galaxies.draw_galaxy()
            lens = self._lens_galaxies.draw_deflector()
            gg_lens = GGLens(deflector_dict=lens, source_dict=source, cosmo=self.cosmo)
            if gg_lens.validity_test(**kwargs_lens_cut):
                return gg_lens

    def get_num_lenses(self):
        return self._lens_galaxies.deflector_number()

    def get_num_sources(self):
        return self._source_galaxies.galaxies_number()

    def get_num_sources_tested_mean(self, testarea):
        """
        Compute the mean of source galaxies needed to be tested within the test area.
        num_sources_tested_mean/ testarea = num_sources/ f_sky;
        testarea is in units of arcsec^2, f_sky is in units of deg^2. 1 deg^2 = 12960000 arcsec^2
        """
        num_sources = self._source_galaxies.galaxies_number()
        num_sources_tested_mean = (testarea * num_sources) / (12960000 * self.f_sky.to_value('deg2'))
        return num_sources_tested_mean

    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """
        Draw a realization of the expected distribution (Poisson) around the mean
        for the number of source galaxies tested.
        """
        if num_sources_tested_mean is None:
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    def draw_population(self, kwargs_lens_cuts):
        """
        return full population list of all lenses within the area
        # TODO: need to implement a version of it. (improve the algorithm)
            Draw a population of galaxy-galaxy lenses within the area.

        :param kwargs_lens_cuts: validity test keywords
        :type kwargs_lens_cuts: dict
        :return: List of GGLens instances with parameters of the deflectors and lens and source light.
        :rtype: list
        """

        # Initialize an empty list to store the GGLens instances
        gg_lens_population = []
        # Estimate the number of lensing systems
        num_lenses = self._lens_galaxies.deflector_number()
        # num_sources = self._source_galaxies.galaxies_number()
        #        print(num_sources_tested_mean)
        #        print("num_lenses is " + str(num_lenses))
        #        print("num_sources is " + str(num_sources))
        #        print(np.int(num_lenses * num_sources_tested_mean))

        # Draw a population of galaxy-galaxy lenses within the area.
        for _ in range(num_lenses):
            lens = self._lens_galaxies.draw_deflector()
            test_area = draw_test_area(deflector=lens)
            num_sources_range = self.get_num_sources_tested(testarea=test_area)
            # TODO: to implement this for a multi-source plane lens system
            if num_sources_range > 0:
                n = 0
                while n < num_sources_range:
                    source = self._source_galaxies.draw_galaxy()
                    gg_lens = GGLens(deflector_dict=lens, source_dict=source, cosmo=self.cosmo, test_area=test_area)
                    # Check the validity of the lens system
                    if gg_lens.validity_test(**kwargs_lens_cuts):
                        gg_lens_population.append(gg_lens)
                        n = num_sources_range
                    else:
                        n += 1
        return gg_lens_population
