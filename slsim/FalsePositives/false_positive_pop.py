from slsim.FalsePositives.false_positive import FalsePositive
from slsim.Lenses.lens_pop import area_theta_e_infinity
from slsim.LOS.los_pop import LOSPop
import random
import numpy as np


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


class FalsePositiveQuasarPop(object):
    """Class to perform samples of false positive Quasar populations."""

    def __init__(
        self,
        elliptical_galaxy_population,
        quasar_population,
        field_galaxy_population=None,
        field_galaxy_number_density=0.05,
        cosmo=None,
        los_pop=None,
        source_number_choice=[2, 4],
        weights_for_source_number=None,
        clustering_mode="random",
        test_area_factor=1,
    ):
        """
        Args:
        :param elliptical_galaxy_population: Deflector population as a deflectors class
         instance.
        :param quasar_population: Source population as a sources class inatnce.
        :param field_galaxy_population: Optional sources class instance for field galaxies.
        :param field_galaxy_number_density: Number density of field galaxies per square arcsecond, used if field_galaxy_population is provided.
        :param cosmo: astropy.cosmology instance
        :param los_pop: LOSPop instance which manages line-of-sight (LOS) effects and Gaussian mixture models in a simulation or analysis context.
        :param source_number_choice: A list of integers to choose source number from. If None, defaults to [2, 4].
        :param weights_for_source_number: A list of weights corresponding to the probabilities of
         selecting each value in source_number_choice. If None, all choices are equally
         likely. Defaults to None.
        :param clustering_mode: 'random' for chance alignments, 'ring' for beads-on-a-string
        """
        self.cosmo = cosmo
        self._lens_galaxies = elliptical_galaxy_population
        self._sources = quasar_population
        self._choice = source_number_choice
        self._weights = weights_for_source_number
        self._clustering_mode = clustering_mode
        self._test_area_factor = test_area_factor
        self.los_config = los_pop or LOSPop()

        self._field_galaxy_population = field_galaxy_population
        self._field_galaxy_number_density = field_galaxy_number_density

    def draw_deflector(self):
        deflector = self._lens_galaxies.draw_deflector()
        z_max = deflector.redshift + 0.002
        return deflector, z_max

    def draw_field_galaxies(self, area, z_max):
        if self._field_galaxy_population is None:
            return []
        expected_number = self._field_galaxy_number_density * area
        number_of_galaxies = np.random.poisson(expected_number)
        field_galaxies = []
        for _ in range(number_of_galaxies):
            galaxy = self._field_galaxy_population.draw_source(
                z_max=z_max
            )  # Assuming high z_max for field galaxies
            if galaxy is not None:
                galaxy.update_center(area=area)
                field_galaxies.append(galaxy)
        return field_galaxies

    def draw_sources(self, z_max, theta_e):
        """Draw quasar source(s) with specific geometric constraints."""
        source_number = random.choices(self._choice, weights=self._weights)[0]
        source_list = []

        for i in range(source_number):
            # 1. Draw a fresh source
            source = self._sources.draw_source()
            if source is None:
                return None

            # 2. Determine Position based on trap geometry
            if self._clustering_mode == "ring":
                r = random.uniform(0.8 * theta_e, 1.2 * theta_e)
                phi = (2 * np.pi * i / source_number) + random.uniform(-0.3, 0.3)
                x_pos = r * np.cos(phi)
                y_pos = r * np.sin(phi)
            else:
                # Random box scaling with Einstein radius
                # We use a box size ~ 3x Einstein radius to get close alignments
                box_limit = max(1.5, theta_e * 3.0)
                x_pos = random.uniform(-box_limit, box_limit)
                y_pos = random.uniform(-box_limit, box_limit)

            # 3. Update the source position
            source.update_center(center_x=x_pos, center_y=y_pos)

            source_list.append(source)

        # Field galaxies
        area = (
            4 * (2 * max(1.5, theta_e * 3.0)) ** 2
        )  # Area of the box * 4 used for random placement
        field_galaxies = self.draw_field_galaxies(area, z_max)
        source_list.extend(field_galaxies)

        return source_list[0] if len(source_list) == 1 else source_list

    def draw_false_positive(self, number=1):
        fp_pop = []
        for _ in range(number):
            successful = False
            while not successful:
                deflector, z_max = self.draw_deflector()

                theta_e = deflector.theta_e_infinity(cosmo=self.cosmo)

                sources = self.draw_sources(z_max, theta_e)
                if sources is None:
                    continue

                fp = FalsePositive(
                    deflector_class=deflector, source_class=sources, cosmo=self.cosmo
                )
                fp_pop.append(fp)
                successful = True
        return fp_pop[0] if number == 1 else fp_pop
