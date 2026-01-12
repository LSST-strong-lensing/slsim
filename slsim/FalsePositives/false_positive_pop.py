from slsim.FalsePositives.false_positive import FalsePositive
from slsim.Lenses.lens_pop import area_theta_e_infinity
from slsim.LOS.los_pop import LOSPop
import random


class FalsePositivePop(object):
    """Class to perform samples of false positive population.

    Here, false positives refer to a configuration that includes an
    elliptical galaxy at the center with blue galaxies surrounding the
    central elliptical galaxy. This class generates specified number of
    false positives.
    """

    def __init__(
        self,
        elliptical_galaxy_population,
        blue_galaxy_population,
        cosmo=None,
        los_pop=None,
        source_number_choice=[1, 2, 3],
        weights_for_source_number=None,
        test_area_factor=1,
    ):
        """
        Args:
        :param elliptical_galaxy_population: Deflector population as a deflectors class
         instance.
        :param blue_galaxy_population: Source population as a sources class inatnce.
        :param cosmo: astropy.cosmology instance
        :param los_pop: LOSPop instance which manages line-of-sight (LOS) effects
         and Gaussian mixture models in a simulation or analysis context.
        :param source_number_choice: A list of integers to choose source number from. If
         None, defaults to [1, 2, 3].
        :param weights_for_source_number: A list of weights corresponding to the probabilities of
         selecting each value in source_number_choice. If None, all choices are equally
         likely. Defaults to None.
        :param test_area_factor: A multiplicative factor of a test_area. A test area is
         computed using a velocity dispersion of a central galaxy and that area is
         multiplied by this factor. A default value is 1.
        """

        self.cosmo = cosmo
        self._lens_galaxies = elliptical_galaxy_population
        self._sources = blue_galaxy_population
        self._choice = source_number_choice
        self._weights = weights_for_source_number
        self._test_area_factor = test_area_factor
        self.los_config = los_pop
        if self.los_config is None:
            self.los_config = LOSPop()

    def draw_deflector(self):
        """Draw and prepare a deflector (lens) with tolerance-based z_max.

        :return: a deflector instance and deflector redshift with
            tolerance added.
        """
        deflector = self._lens_galaxies.draw_deflector()
        z_max = deflector.redshift + 0.002  # Adding tolerance to redshift
        return deflector, z_max

    def draw_sources(self, z_max, area=None):
        """Draw source(s) within the redshift limit of z_max.

        :param z_max: maximum redshift for drawn source.
        :param area: area to draw source coordinates, if None, does not
            draw it
        :return: A Source instance or a list of Source instance.
        """
        source_number = random.choices(self._choice, weights=self._weights)[0]
        source_list = []

        for _ in range(source_number):
            source = self._sources.draw_source(z_max=z_max)
            # If no source is available, return None
            if source is None:
                return None
            if area is not None:
                source.update_center(area=area)
            source_list.append(source)
        if source_number == 1:
            sources = source_list[0]
        else:
            sources = source_list
        return sources

    def draw_false_positive(self, number=1):
        """Draw given number of false positives within the cuts of the lens and
        source.

        :param number: number of false positive requested. The default
            value is 1.
        :return: list of FalsePositive() instance.
        """
        false_positive_population = []

        for _ in range(number):
            successful = False
            while not successful:
                # Step 1: Draw deflector
                deflector, z_max = self.draw_deflector()
                # Step 2: Draw sources
                theta_e_infinity = deflector.theta_e_infinity(cosmo=self.cosmo)
                test_area = self._test_area_factor * area_theta_e_infinity(
                    theta_e_infinity=theta_e_infinity
                )
                source = self.draw_sources(z_max, area=test_area)
                if source is None:
                    continue  # Retry if sources are invalid

                # Step 3: Create false positive

                false_positive = FalsePositive(
                    deflector_class=deflector,
                    source_class=source,
                    cosmo=self.cosmo,
                )
                false_positive_population.append(false_positive)
                successful = True
        return (
            false_positive_population[0] if number == 1 else false_positive_population
        )
