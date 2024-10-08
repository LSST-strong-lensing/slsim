import numpy as np

from slsim.false_positive import FalsePositive
from typing import Optional
from astropy.cosmology import Cosmology
from slsim.lens import theta_e_when_source_infinity
from slsim.Sources.source_pop_base import SourcePopBase
from slsim.ParamDistributions.los_config import LOSConfig
from slsim.Deflectors.deflectors_base import DeflectorsBase
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
        """Draw given number of false positive within the cuts of the lens and source.
        :return: list of FalsePositive() instance with parameters of a pair of elliptical
         and blue galaxy.
        """
        false_positive_population = []
        for _ in range(number):
            lens = self._lens_galaxies.draw_deflector()
            tolerance = 0.002
            z_min=lens["z"] + tolerance
            source = self._sources.draw_source(z_min=z_min)
            test_area = draw_test_area(deflector=lens)
            false_positive = FalsePositive(
                deflector_dict=lens,
                source_dict=source,
                deflector_type=self._lens_galaxies.deflector_profile,
                cosmo=self.cosmo,
                source_type=self._sources.source_type,
                light_profile=self._sources.light_profile,
                test_area=test_area
            )
            false_positive_population.append(false_positive)
        if number == 1:
            return false_positive_population[0]
        else:
            return false_positive_population