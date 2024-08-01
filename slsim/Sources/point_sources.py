import numpy.random as random
from slsim.Sources.source_pop_base import SourcePopBase
import warnings
from slsim.selection import deflector_cut


class PointSources(SourcePopBase):
    """Class to describe point sources."""

    def __init__(
        self,
        point_source_list,
        cosmo,
        sky_area,
        kwargs_cut,
        variability_model=None,
        kwargs_variability_model=None,
        light_profile=None,
        list_type="astropy_table",
    ):
        """

        :param point_source_list: list of dictionary with quasar parameters or astropy
         table.
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max. These are 
         the arguments that go into the deflector_cut() definition which is a general 
         defination for performing given cuts in given catalog. For the supernovae 
         sample, we can only apply redshift cuts because supernovae sample contains only
         redshift in this stage.
        :type kwargs_cut: dict
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         point source.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual point_source.
        :param light_profile: keyword for number of sersic profile to use in source
         light model. Always None for this class.
        :param list_type: type of the format of the source catalog. It should be either
         astropy_table or list of astropy table.
        """
        self.n = len(point_source_list)
        self.light_profile = light_profile
        if self.light_profile is not None:
            warning_msg = (
                "The provided light profile %s is not used to describe the point "
                "source. The relevant light profile is None." % light_profile
            )
            warnings.warn(warning_msg, category=UserWarning, stacklevel=2)
        # make cuts
        self._point_source_select = deflector_cut(
            point_source_list, list_type=list_type, **kwargs_cut
        )

        self._num_select = len(self._point_source_select)
        super(PointSources, self).__init__(
            cosmo=cosmo,
            sky_area=sky_area,
            variability_model=variability_model,
            kwargs_variability_model=kwargs_variability_model,
        )

    @property
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        number = self.n
        return number

    @property
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        return self._num_select

    def draw_source(self):
        """Choose source at random with the selected range.

        :return: dictionary of source
        """

        index = random.randint(0, self._num_select - 1)
        point_source = self._point_source_select[index]

        return point_source
