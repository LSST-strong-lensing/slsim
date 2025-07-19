from slsim.Sources.SourceTypes.point_source import PointSource
from slsim.Sources.SourceTypes.extended_source import ExtendedSource


class PointPlusExtendedSource(PointSource, ExtendedSource):
    """Class to manage a single point source and a single extended source
    (host)"""

    def __init__(
        self,
        source_dict,
        extendedsource_type,
        pointsource_type,
        cosmo=None,
        pointsource_kwargs={},
        extendedsource_kwargs={},
    ):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, Interpolated classes, Supernova, and Quasar class.
        :type source_dict: dict or astropy.table.Table .
         eg of a supernova plus host galaxy dict: {"z": 0.8, "mag_i": 22, "n_sersic": 1,
           "angular_size": 0.10, "e1": 0.002, "e2": 0.001, "ra_off": 0.001, "dec_off": 0.005}
        :param extendedsource_type: keyword for specifying light profile model.
        :type extendedsource_type: str. supported types are "single_sersic",
         "double_sersic", "interpolated".
        :param pointsource_type: keyword for specifying point source type.
        :type pointsource_type: str. supported types are "supernova", "quasar", "general_lightcurve".
        :param cosmo: astropy.cosmology instance
        :param pointsource_kwargs: dictionary of keyword arguments for a PointSource.
         For supernova kwargs dict, please see documentation of Supernova class.
         For quasar kwargs dict, please see documentation of Quasar class.
        :param extendedsource_kwargs: dictionary of keyword arguments for ExtendedSource.
         Please see documentation of ExtendedSource() class as well as specific extended source classes.
        """
        # Initialize the extended source. Here, source_dict will contain both host
        # galaxy and point source information but only extended source properties will
        # be read in the ExtendedSource class and only point source properties will be
        # read in the point source class.
        ExtendedSource.__init__(
            self,
            source_dict=source_dict,
            extendedsource_type=extendedsource_type,
            cosmo=cosmo,
            extendedsource_kwargs=extendedsource_kwargs,
        )

        # Initialize the point source.
        PointSource.__init__(
            self,
            source_dict=source_dict,
            pointsource_type=pointsource_type,
            cosmo=cosmo,
            pointsource_kwargs=pointsource_kwargs,
        )
