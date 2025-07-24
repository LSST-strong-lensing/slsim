from abc import ABC, abstractmethod


class SourcePopBase(ABC):
    """Base class with functions all source classes must have to be able to
    render populations."""

    def __init__(
        self,
        cosmo,
        sky_area,
    ):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self.source_type = None
        self.sky_area = sky_area
        self._cosmo = cosmo
        # These quantities are defined here because Source class these quantities and
        # None act as default values.
        self.pointsource_type = None
        self.extendedsource_type = None

    @property
    @abstractmethod
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        pass

    @property
    @abstractmethod
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        pass

    @abstractmethod
    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """
        pass

    @property
    def variability_model(self):
        """
        :return: keyword for the variability model
        """
        return self._variab_model

    @property
    def kwargs_variability(self):
        """
        :return: dict of keyword arguments for the variability model.
        """
        return self._kwargs_variab_model
