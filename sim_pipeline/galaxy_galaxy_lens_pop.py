from sim_pipeline.Pipelines.skypy_pipeline import SkyPyPipeline
from sim_pipeline.galaxy_galaxy_lens import (
    GalaxyGalaxyLens,
    theta_e_when_source_infinity,
)
import numpy as np
from sim_pipeline.lensed_sample import LensedSample


class GalaxyGalaxyLensPop(LensedSample):
    """Class to perform samples of galaxy-galaxy lensing."""

    def __init__(
        self,
        deflector_type="elliptical",
        source_type="galaxies",
        kwargs_deflector_cut=None,
        kwargs_source_cut=None,
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=None,
        filters=None,
        cosmo=None,
    ):
        """

        :param deflector_type: type of the lens
        :type deflector_type: string
        :param source_type: type of the source
        :type source_type: string
        :param kwargs_deflector_cut: cuts on the deflector to be excluded in the sample
        :type kwargs_deflector_cut: dict
        :param kwargs_source_cut: cuts on the source to be excluded in the sample
        :type kwargs_source_cut: dict
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param filters: filters for SED integration
        :type filters: list of strings or None
        """
        super().__init__(sky_area, cosmo)
        if deflector_type in ["elliptical", "all-galaxies"] or source_type in [
            "galaxies"
        ]:
            pipeline = SkyPyPipeline(
                skypy_config=skypy_config, sky_area=sky_area, filters=filters
            )
        if kwargs_deflector_cut is None:
            kwargs_deflector_cut = {}
        if kwargs_mass2light is None:
            kwargs_mass2light = {}

        if deflector_type == "elliptical":
            from sim_pipeline.Deflectors.elliptical_lens_galaxies import (
                EllipticalLensGalaxies,
            )

            self._lens_galaxies = EllipticalLensGalaxies(
                pipeline.red_galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

        elif deflector_type == "all-galaxies":
            from sim_pipeline.Deflectors.all_lens_galaxies import AllLensGalaxies

            red_galaxy_list = pipeline.red_galaxies
            blue_galaxy_list = pipeline.blue_galaxies

            self._lens_galaxies = AllLensGalaxies(
                red_galaxy_list = red_galaxy_list,
                blue_galaxy_list= blue_galaxy_list,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

        else:
            raise ValueError("deflector_type %s is not supported" % deflector_type)

        if kwargs_source_cut is None:
            kwargs_source_cut = {}
        if source_type == "galaxies":
            from sim_pipeline.Sources.galaxies import Galaxies

            self._sources = Galaxies(
                pipeline.blue_galaxies,
                kwargs_cut=kwargs_source_cut,
                cosmo=cosmo,
                sky_area=sky_area,
            )
            self._source_model_type = "extended"
        elif source_type == "quasars":
            from sim_pipeline.Sources.quasars import Quasars

            self._sources = Quasars(cosmo=cosmo, sky_area=sky_area)
            self._source_model_type = "point_source"
        else:
            raise ValueError("source_type %s is not supported" % source_type)
        self.cosmo = cosmo
        self.f_sky = sky_area

    def generate_random_lensed_system(self, **kwargs_lens_cut):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        # TODO: make sure mass function is preserved, # as well as option to draw all
        lenses within the cuts within the area

        :return: GalaxyGalaxyLens() instance with parameters of the deflector and lens
            and source light
        """
        while True:
            source = self._sources.draw_source()
            lens = self._lens_galaxies.draw_deflector()
            gg_lens = GalaxyGalaxyLens(
                deflector_dict=lens,
                source_dict=source,
                cosmo=self.cosmo,
                source_type=self._source_model_type,
            )
            if gg_lens.validity_test(**kwargs_lens_cut):
                return gg_lens

    def potential_deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that are being
        considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        return len(self._lens_galaxies)

    def potential_source_number(self):
        """Number of sources that are being considered to be placed in the sky area
        potentially aligned behind deflectors.

        :return: number of potential sources
        """
        return len(self._sources)

    def get_num_sources_tested_mean(self, testarea):
        """Compute the mean of source galaxies needed to be tested within the test area.

        num_sources_tested_mean/ testarea = num_sources/ f_sky; testarea is in units of
        arcsec^2, f_sky is in units of deg^2. 1 deg^2 = 12960000 arcsec^2
        """
        num_sources = len(self._sources)
        num_sources_tested_mean = (testarea * num_sources) / (
            12960000 * self.f_sky.to_value("deg2")
        )
        return num_sources_tested_mean

    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """Draw a realization of the expected distribution (Poisson) around the mean for
        the number of source galaxies tested."""
        if num_sources_tested_mean is None:
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    def draw_sample(self, kwargs_lens_cuts):
        """Return full population list of all lenses within the area # TODO: need to
        implement a version of it. (improve the algorithm)

        :param kwargs_lens_cuts: validity test keywords
        :type kwargs_lens_cuts: dict
        :return: List of GalaxyGalaxyLens instances with parameters of the deflectors
            and lens and source light.
        :rtype: list
        """

        # Initialize an empty list to store the GalaxyGalaxyLens instances
        gg_lens_population = []
        # Estimate the number of lensing systems
        num_lenses = len(self._lens_galaxies)
        # num_sources = self._source_galaxies.galaxies_number()
        #        print(num_sources_tested_mean)
        #        print("num_lenses is " + str(num_lenses))
        #        print("num_sources is " + str(num_sources))
        #        print(np.int(num_lenses * num_sources_tested_mean))

        # Draw a population of galaxy-galaxy lenses within the area.
        for _ in range(num_lenses):
            lens = self._lens_galaxies.draw_deflector()
            test_area = draw_test_area(deflector=lens)
            num_sources_tested = self.get_num_sources_tested(testarea=test_area)
            # TODO: to implement this for a multi-source plane lens system
            if num_sources_tested > 0:
                n = 0
                while n < num_sources_tested:
                    source = self._sources.draw_source()
                    gg_lens = GalaxyGalaxyLens(
                        deflector_dict=lens,
                        source_dict=source,
                        cosmo=self.cosmo,
                        test_area=test_area,
                        source_type=self._source_model_type,
                    )
                    # Check the validity of the lens system
                    if gg_lens.validity_test(**kwargs_lens_cuts):
                        gg_lens_population.append(gg_lens)
                        # if a lens system passes the validity test, code should exit
                        # the loop. so, n should be greater or equal to
                        # num_sources_tested which will break the while loop
                        # (instead of this one can simply use break).
                        n = num_sources_tested
                    else:
                        n += 1
        return gg_lens_population


def draw_test_area(deflector):
    """Draw a test area around the deflector.

    :param deflector: deflector dictionary
    :return: test area in arcsec^2
    """
    theta_e_infinity = theta_e_when_source_infinity(deflector)
    test_area = np.pi * (theta_e_infinity * 2.5) ** 2
    return test_area
