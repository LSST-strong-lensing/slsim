from slsim.Sources.source_pop_base import SourcePopBase
from slsim.Sources.galaxies import Galaxies
from slsim.selection import object_cut


class PointPlusExtendedSources(Galaxies, SourcePopBase):
    """Class to describe point and extended sources."""

    def __init__(
        self,
        point_plus_extended_sources_list,
        cosmo,
        sky_area,
        kwargs_cut,
        variability_model=None,
        kwargs_variability_model=None,
        light_profile="single_sersic",
        list_type="astropy_table",
        catalog_type=None,
    ):
        """

        :param point_plus_extended_sources_list: list of dictionary with point and
         extended source parameters or astropy table of sources.
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
        :param light_profile: keyword for number of sersic profile to use in source
         light model. accepted kewords: "single_sersic", "double_sersic".
        :param list_type: format of the source catalog file. Currently, it supports
         a single astropy table or a list of astropy tables.
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it.
        :type catalog_type: str. eg: "scotch" or None
        """

        object_list = object_cut(
            point_plus_extended_sources_list,
            list_type=list_type,
            object_type="point",
            **kwargs_cut
        )
        Galaxies.__init__(
            self,
            galaxy_list=object_list,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut={},
            light_profile=light_profile,
            list_type=list_type,
            catalog_type=catalog_type,
        )
        SourcePopBase.__init__(
            self, cosmo, sky_area, variability_model, kwargs_variability_model
        )
        self.source_type = "point_plus_extended"
