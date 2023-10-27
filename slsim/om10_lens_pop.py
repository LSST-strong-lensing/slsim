from slsim.lensed_population_base import LensedPopulationBase


class OM10LensPop(LensedPopulationBase):
    """Abstract Base Class to create a sample of lensed systems.

    All object that inherit from Lensed Sample must contain the methods it contains.
    """

    def __init__(self, sky_area=None, cosmo=None):
        """
        :param source_type: type of the source
        :type source_type: string
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :type sky_area: `~astropy.units.Quantity`
        """

    def select_lens_at_random(self):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        # as well as option to draw all lenses within the cuts within the area

        :return: OM10Lens() instance with parameters of the deflector and lens and
            source light
        """
        return

    def deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that are being
        considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        pass

    def source_number(self):
        """Number of sources that are being considered to be placed in the sky area
        potentially aligned behind deflectors.

        :return: number of sources
        """
        pass

    def draw_population(self):
        """Return full sample list of all lenses within the area.

        :return: List of LensedSystem instances with parameters of the deflectors and
            source.
        :rtype: list
        """

        pass
