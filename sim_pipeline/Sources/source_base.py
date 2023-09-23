<<<<<<< HEAD
from abc import ABC, abstractmethod

class SourceBase(ABC):
    """
    Base class with functions all source classes must have to be able to render populations
    """
=======
class SourceBase(object):
    """Base class with functions all source classes must have to be able to render
    populations."""

>>>>>>> main
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
<<<<<<< HEAD
        pass
    
    @abstractmethod
=======
        raise NotImplementedError(
            "Function source_number not implemented in chosen source class."
        )

>>>>>>> main
    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """
<<<<<<< HEAD
        pass
=======
        raise NotImplementedError(
            "Function draw_source not implemented in chosen source class."
        )
>>>>>>> main
