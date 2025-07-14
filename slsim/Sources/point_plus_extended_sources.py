from slsim.Sources.source_pop_base import SourcePopBase
from slsim.Sources.source import Source
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
        list_type="astropy_table",
        catalog_type=None,
        source_size="Bernadi",
        pointsource_type=None,
        extendedsource_type=None,
        pointsource_kwargs={},
        extendedsource_kwargs={},
    ):
        """

        :param point_plus_extended_sources_list: list of dictionary with point and
         extended source parameters or astropy table of sources.
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param list_type: format of the source catalog file. Currently, it supports
         a single astropy table or a list of astropy tables.
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it.
        :type catalog_type: str. eg: "scotch" or None
        :param source_size: If "Bernardi", computes galaxy size using g-band
         magnitude otherwise rescales skypy source size to Shibuya et al. (2015):
         https://iopscience.iop.org/article/10.1088/0067-0049/219/2/15/pdf
        :param pointsource_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :param extendedsource_type: keyword for number of sersic profile to use in source
         light model. accepted kewords: "single_sersic", "double_sersic".
        :param pointsource_kwargs: dictionary of keyword arguments for PointSource.
         For supernova kwargs dict, please see documentation of SupernovaEvent class.
         For quasar kwargs dict, please see documentation of Quasar class.
         Eg of supernova kwargs: pointsource_kwargs={
         "variability_model": "light_curve", "kwargs_variability": ["supernovae_lightcurve",
            "i", "r"], "sn_type": "Ia", "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab", "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None}.
        :param extendedsource_kwargs: dictionary of keyword arguments for ExtendedSource.
         Please see documentation of ExtendedSource() class as well as specific extended source classes.
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
            list_type=list_type,
            catalog_type=catalog_type,
            source_size=source_size,
            extendedsource_type=extendedsource_type,
            extendedsource_kwargs=extendedsource_kwargs,
        )
        SourcePopBase.__init__(
            self,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        self.source_type = "point_plus_extended"
        self.pointsource_kwargs = pointsource_kwargs
        self.pointsource_type = pointsource_type

    def draw_source(self, z_max=None):
        """Choose source at random.

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :return: instance of Source class
        """
        galaxy = self.draw_source_dict(z_max)
        if galaxy is None:
            return None
        source_class = Source(
            source_dict=galaxy,
            source_type=self.source_type,
            cosmo=self._cosmo,
            extendedsource_type=self.light_profile,
            pointsource_type=self.pointsource_type,
            pointsource_kwargs=self.pointsource_kwargs,
            extendedsource_kwargs=self.extendedsource_kwargs,
        )
        return source_class
