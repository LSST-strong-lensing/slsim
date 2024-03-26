import numpy.random as random
from slsim.Sources.source_pop_base import SourcePopBase
import warnings


class Quasars(SourcePopBase):
    """Class to describe quasars as sources."""

    def __init__(
        self,
        quasar_list,
        cosmo,
        sky_area,
        variability_model=None,
        kwargs_variability_model=None,
        light_profile=None,
    ):
        """

        :param quasar_list: list of dictionary with quasar parameters
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         quasars.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual quasars.
        :param light_profile: keyword for number of sersic profile to use in source
         light model. Always None for this class.
        """
        self.n = len(quasar_list)
        self.light_profile = light_profile
        if self.light_profile is not None:
            warning_msg = (
                "The provided light profile %s is not used to describe the point "
                "source. The relevant light profile is None." % light_profile
            )
            warnings.warn(warning_msg, category=UserWarning, stacklevel=2)
        # make cuts
        self._quasar_select = quasar_list  # can apply a filter here

        self._num_select = len(self._quasar_select)
        super(Quasars, self).__init__(
            cosmo=cosmo,
            sky_area=sky_area,
            variability_model=variability_model,
            kwargs_variability_model=kwargs_variability_model,
        )

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
