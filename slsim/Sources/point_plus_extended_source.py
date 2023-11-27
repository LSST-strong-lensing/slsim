from slsim.Sources.source_pop_base import SourcePopBase
from slsim.Sources.galaxies import Galaxies


class PointPlusExtendedSource(Galaxies, SourcePopBase):
    """Class to describe point and extended sources."""

    def __init__(
        self,
        point_plus_extended_source_list,
        cosmo,
        sky_area,
        kwargs_cut,
        variability_model=None,
        kwargs_variability_model=None,
    ):
        """

        :param pes_list: list of dictionary with point and extended source parameters
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         source.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual source.
        """
        Galaxies.__init__(
            self, point_plus_extended_source_list, kwargs_cut, cosmo, sky_area
        )
        SourcePopBase.__init__(
            self, cosmo, sky_area, variability_model, kwargs_variability_model
        )
