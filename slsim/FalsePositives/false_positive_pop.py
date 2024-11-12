from slsim.FalsePositives.false_positive import FalsePositive
from typing import Optional
from astropy.cosmology import Cosmology
from slsim.Sources.source_pop_base import SourcePopBase
from slsim.ParamDistributions.los_config import LOSConfig
from slsim.Deflectors.deflectors_base import DeflectorsBase
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.lens_pop import draw_test_area
import random


class FalsePositivePop(object):
    """Class to perform samples of false positive population. Here, Here, false 
    positives refer to a configuration that includes an elliptical galaxy at the center 
    with blue galaxies surrounding the central elliptical galaxy. This class generates 
    specified number of false positives."""

    def __init__(
        self,
        elliptical_galaxy_population: DeflectorsBase,
        blue_galaxy_population: SourcePopBase,
        cosmo: Optional[Cosmology] = None,
        los_config: Optional[LOSConfig] = None,
        source_number_choice: Optional[list[int]] = [1, 2, 3],
        weights_for_source_number: Optional[list[int]] = None,
    ):
        """
        Args:
            deflector_population (DeflectorsBase): Deflector population as an instance of a DeflectorsBase subclass.
            source_population (SourcePopBase): Source population as an instance of a SourcePopBase subclass
            cosmo (Optional[Cosmology], optional): AstroPy Cosmology instance. If None, defaults to flat LCDM with h0=0.7 and Om0=0.3.
                                                   Defaults to None.
            los_config (Optional[LOSConfig], optional): Configuration for line of sight distribution. Defaults to None.
            source_number_choice (Optional[list[int]], optional): A list of integers to choose source number from. If None, defaults to [1, 2, 3].
            weights (Optional[list[int]], optional): A list of weights corresponding to the probabilities of selecting each value
                                             in source_number_choice. If None, all choices are equally likely.
                                             Defaults to None.
        """

        self.cosmo = cosmo
        self._lens_galaxies = elliptical_galaxy_population
        self._sources = blue_galaxy_population
        self._choice = source_number_choice
        self._weights = weights_for_source_number
        self.los_config = los_config
        if self.los_config is None:
            self.los_config = LOSConfig()
    
    def draw_deflector(self):
        """Draw and prepare a deflector (lens) with tolerance-based z_max.
        
        :return: a deflector instance and deflector redshift with tolerance added.
        """
        lens = self._lens_galaxies.draw_deflector()
        deflector = Deflector(
            deflector_type=self._lens_galaxies.deflector_profile,
            deflector_dict=lens,
        )
        z_max = deflector.redshift + 0.002  # Adding tolerance to redshift
        return deflector, z_max

    def draw_sources(self, z_max):
        """Draw source(s) within the redshift limit of z_max.
        :param z_max: maximum redshift for drawn source.
        :return: A Source instance or a list of Source instance.

        """
        source_number = random.choices(self._choice, weights=self._weights)[0]
        source_list = []

        for _ in range(source_number):
            source = self._sources.draw_source(z_max=z_max)
            # If no source is available, return None
            if source is None:
                return None
            source_list.append(
                Source(
                    source_dict=source,
                    cosmo=self.cosmo,
                    source_type=self._sources.source_type,
                    light_profile=self._sources.light_profile,
                )
            )
        if source_number==1:
            sources = source_list[0]
        else:
            sources = source_list
        return sources

    def draw_false_positive(self, number=1):
        
        """Draw given number of false positives within the cuts of the lens and source.

        :param number: number of false positive requested. The default value is 1.
        :return: list of FalsePositive() instance.
         """
        false_positive_population = []

        for _ in range(number):
            successful = False
            while not successful:
                # Step 1: Draw deflector
                lens, z_max = self.draw_deflector()
                # Step 2: Draw sources
                source = self.draw_sources(z_max)
                if source is None:
                    continue  # Retry if sources are invalid

                # Step 3: Create false positive
                vd=lens.velocity_dispersion(cosmo=self.cosmo)
                test_area = 3 * draw_test_area(
                    v_sigma=vd)
                false_positive = FalsePositive(
                            deflector_class=lens,
                            source_class=source,
                            cosmo=self.cosmo,
                            test_area=test_area,
                        )
                false_positive_population.append(false_positive)
                successful = True
        return false_positive_population[0] if number==1 else false_positive_population
