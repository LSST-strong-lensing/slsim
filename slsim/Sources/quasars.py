import numpy.random as random
from slsim.Sources.source_base import SourceBase


class Quasars(SourceBase):
    """Class to describe quasars as sources."""

    def __init__(
        self,
        quasar_list,
        cosmo,
        sky_area,
        variability_model=None,
        kwargs_variability_model=None,
    ):
        """

        :param quasar_list: list of dictionary with quasar parameters
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self.n = len(quasar_list)
        self._variab_model = variability_model
        self._kwargs_variab_model = kwargs_variability_model
        # make cuts
        self._quasar_select = quasar_list  # can apply a filter here

        self._num_select = len(self._quasar_select)
        super(Quasars, self).__init__(cosmo=cosmo, sky_area=sky_area)

    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        number = self.n
        return number

    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """

        index = random.randint(0, self._num_select - 1)
        quasar = self._quasar_select[index]

        return quasar

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
