from abc import ABC, abstractmethod

class SourceBase(ABC):
    """
    Base class with functions all source classes must have to be able to render populations
    """
    def __init__(self, cosmo, sky_area):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self._cosmo = cosmo
        self._sky_area = sky_area

    @abstractmethod
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        pass
    
    @abstractmethod
    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """
        pass
