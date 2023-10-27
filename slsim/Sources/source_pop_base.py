from abc import ABC, abstractmethod


class SourcePopBase(ABC):
    """Base class with functions all source classes must have to be able to render
    populations."""

    def __init__(
        self, cosmo, sky_area, variability_model=None, kwargs_variability_model=None
    ):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         sources.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual sources.
        """
        self._cosmo = cosmo
        self._sky_area = sky_area
        self._variab_model = variability_model
        self._kwargs_variab_model = kwargs_variability_model

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
