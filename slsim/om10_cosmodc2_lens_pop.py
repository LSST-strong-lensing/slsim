from slsim.lensed_population_base import LensedPopulationBase
from slsim.Pipelines.catalog_pipeline import catalogPipeline
from slsim.om10_cosmodc2_lens import (
    OM10LensSystem,
)
import numpy as np


class OM10LensPop(LensedPopulationBase):
    """Abstract Base Class to create a sample of lensed systems.

    All object that inherit from Lensed Sample must contain the methods it contains.
    """

    def __init__(self, sky_area=None, cosmo=None, catalog_config="data/OM10"):
        """
        :param source_type: type of the source
        :type source_type: string
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :type sky_area: `~astropy.units.Quantity`
        """
        super().__init__(sky_area, cosmo)
        from slsim.Deflectors.om10_lens_galaxies import (
            OM10Lens,
        )

        self._pipeline = catalogPipeline(catalog_config=catalog_config)

        self._lens_galaxies = OM10Lens(
            deflector_input=self._pipeline.deflectors,
            kwargs_cut=None,
            cosmo=cosmo,
            sky_area=sky_area,
        )

        from slsim.Sources.cosmoDC2AGN import (
            cosmoDC2AGN,
        )

        self._source_quasars = cosmoDC2AGN(
            source_input=self._pipeline.sources, cosmo=cosmo, sky_area=sky_area
        )
        self.cosmo = cosmo

    def select_lens_at_random(self):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        # as well as option to draw all lenses within the cuts within the area

        :return: OM10Lens() instance with parameters of the deflector and lens and
            source light
        """
        index = np.random.choice(self._lens_galaxies.deflector_table.index)
        source = self._source_quasars.draw_source(index)
        lens = self._lens_galaxies.draw_deflector(index)
        lens_system = OM10LensSystem(
            deflector_dict=lens,
            source_dict=source,
            cosmo=self.cosmo,
        )
        return lens_system

    def select_specific_lens(self, index):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        # as well as option to draw all lenses within the cuts within the area

        :return: OM10Lens() instance with parameters of the deflector and lens and
            source light
        """
        source = self._source_quasars.draw_source(index)
        lens = self._lens_galaxies.draw_deflector(index)
        lens_system = OM10LensSystem(
            deflector_dict=lens,
            source_dict=source,
            cosmo=self.cosmo,
        )
        return lens_system

    def deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that are being
        considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        return self._lens_galaxies.deflector_number

    def source_number(self):
        """Number of sources that are being considered to be placed in the sky area
        potentially aligned behind deflectors.

        :return: number of sources
        """
        return self._source_quasars.source_number

    def draw_population(self):
        """Return full sample list of all lenses within the area.

        :return: List of LensedSystem instances with parameters of the deflectors and
            source.
        :rtype: list
        """
        lens_population = []
        for i in self._pipeline.deflectors.index:
            lens_population.append(self.select_specific_lens(i))

        return lens_population
