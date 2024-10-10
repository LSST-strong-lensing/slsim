from slsim.false_positive import FalsePositive
from typing import Optional
from astropy.cosmology import Cosmology
from slsim.Sources.source_pop_base import SourcePopBase
from slsim.ParamDistributions.los_config import LOSConfig
from slsim.Deflectors.deflectors_base import DeflectorsBase
from slsim.Sources.source import Source
from slsim.Deflectors.deflector import Deflector
from slsim.lens_pop import draw_test_area


class FalsePositivePop(object):
    """Class to perform samples of false positive population."""

    def __init__(
        self,
        elliptical_galaxy_population: DeflectorsBase,
        blue_galaxy_population: SourcePopBase,
        cosmo: Optional[Cosmology] = None,
        los_config: Optional[LOSConfig] = None,
    ):
        """
        Args:
            deflector_population (DeflectorsBase): Deflector population as an instance of a DeflectorsBase subclass.
            source_population (SourcePopBase): Source population as an instance of a SourcePopBase subclass
            cosmo (Optional[Cosmology], optional): AstroPy Cosmology instance. If None, defaults to flat LCDM with h0=0.7 and Om0=0.3.
                                                   Defaults to None.
            los_config (Optional[LOSConfig], optional): Configuration for line of sight distribution. Defaults to None.
        """

        self.cosmo = cosmo
        self._lens_galaxies = elliptical_galaxy_population
        self._sources = blue_galaxy_population
        self.los_config = los_config
        if self.los_config is None:
            self.los_config = LOSConfig()

    def draw_false_positive(self, number=1):
        """Draw given number of false positives within the cuts of the lens and source.

        :param number: number of false positive requested. The default value is 1.
        :return: list of FalsePositive() instance with parameters of a pair of 
         elliptical and blue galaxy.
        """
        false_positive_population = []
        
        for _ in range(number):
            successful = False
            while not successful:
                try:
                    # Sample a lens (deflector)
                    lens = self._lens_galaxies.draw_deflector()
                    _lens = Deflector(
                        deflector_type=self._lens_galaxies.deflector_profile,
                        deflector_dict=lens)
                    tolerance = 0.002
                    z_max = _lens.redshift + tolerance
                    # Try to draw a source with the z_max based on the lens redshift
                    source = self._sources.draw_source(z_max=z_max)
                    _source = Source(
                        source_dict=source,
                        cosmo=self.cosmo,
                        source_type=self._sources.source_type,
                        light_profile=self._sources.light_profile)
                    # Compute test area for false positive position. 
                    # This area will be used to determine the position of false positive.
                    test_area = 3 * draw_test_area(deflector=lens)

                    # Create a FalsePositive instance with the lens and source information
                    false_positive = FalsePositive(
                        deflector_class=_lens,
                        source_class=_source,
                        cosmo=self.cosmo,
                        test_area=test_area
                    )

                    # Add the false positive to the population
                    false_positive_population.append(false_positive)
                    successful = True 

                except ValueError as e:
                    # Handle the specific case where no sources are found for z_max
                    if str(e).startswith("No galaxies found"):
                        continue
        if number == 1:
            return false_positive_population[0]
        else:
            return false_positive_population
