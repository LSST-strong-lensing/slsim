from slsim.Sources.SourceTypes.point_plus_extended_source import PointPlusExtendedSource

_SUPPORTED_POINT_SOURCES = ["supernova", "quasar", "general_lightcurve"]
_SUPPORTED_EXTENDED_SOURCES = [
    "single_sersic",
    "double_sersic",
    "catalog_source",
    "interpolated",
]


class Source(object):
    """Class to manage an individual source."""

    def __init__(
        self,
        extended_source_type=None,
        point_source_type=None,
        **source_dict,
    ):
        """
        :param source_type: Keyword to specify type of the source. Supported source types are
         'extended', 'point_source', and 'point_plus_extended' supported
        :type source_type: str
        :param source_dict: Source properties. Can be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the individual classes, such as SingleSersic, DoubleSersic, Interpolated classes, SupernovaEvent,
         and Quasar class.
        :type source_dict: dict or astropy.table.Table .

        """
        print(extended_source_type, point_source_type, "test source types")
        self.extended_source_type = extended_source_type
        self.point_source_type = point_source_type
        if extended_source_type is not None and point_source_type is not None:
            source_type = "point_plus_extended"
            self.source_type = source_type
        elif extended_source_type is not None:
            source_type = extended_source_type
            self.source_type = "extended"
        elif point_source_type is not None:
            source_type = point_source_type
            self.source_type = "point_source"
        else:
            raise ValueError("either extended_source_type of point_source_type need to set.")

        # point sources
        if source_type in ["supernova"]:
            from slsim.Sources.SourceTypes.supernova_event import SupernovaEvent
            self._source = SupernovaEvent(**source_dict)
        elif source_type in ["quasar"]:
            from slsim.Sources.SourceTypes.quasar import Quasar
            self._source = Quasar(**source_dict)
        elif source_type in ["general_lightcurve"]:
            from slsim.Sources.SourceTypes.general_lightcurve import GeneralLightCurve
            self._source = GeneralLightCurve(**source_dict)

        # extended sources
        elif source_type in ["single_sersic"]:
            from slsim.Sources.SourceTypes.single_sersic import SingleSersic
            self._source = SingleSersic(**source_dict)
        elif source_type in ["double_sersic"]:
            from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
            self._source = DoubleSersic(**source_dict)
        elif source_type in ["catalog_source"]:
            from slsim.Sources.SourceTypes.catalog_source import CatalogSource
            self._source = CatalogSource(**source_dict)
        elif source_type in ["interpolated"]:
            from slsim.Sources.SourceTypes.interpolated_image import Interpolated
            self._source = Interpolated(**source_dict)

        # point source plus extended source
        elif source_type in ["point_plus_extended"]:
            self._source = PointPlusExtendedSource(
                extended_source_type=extended_source_type,
                point_source_type=point_source_type,
                **source_dict
            )
        else:
            raise ValueError(
                "source type %s not supported. Chose among %s for extended sources and %s for point sources."
                % (source_type, _SUPPORTED_EXTENDED_SOURCES, _SUPPORTED_POINT_SOURCES)
            )

    @property
    def name(self):
        """
        meaningful name string of the source

        :return: name string
        """
        return self._source.name

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._source.redshift

    @property
    def angular_size(self):
        """Returns angular size of the extended source."""

        return self._source.angular_size

    @property
    def ellipticity(self):
        """Returns ellipticity components of extended source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return self._source.ellipticity

    def extended_source_position(self, reference_position=None, draw_area=None):
        """Extended source position. If a center has already been provided (and
        stored in the source_dict), then it is simply returned. Otherwise, a
        source position is drawn uniformly within the circle of the test area
        centered on the deflector position.

        :param reference_position: reference position. the source position will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a source
         position. The default choice is None. In this case
         source_dict must contain source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._source.extended_source_position(
            reference_position, draw_area
        )

    def point_source_position(self, reference_position=None, draw_area=None):
        """Point source position. point source could be at the center of the
        extended source, or it can be off from center of the extended source.

        :param reference_position: reference position. the source position will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case
         source_dict must contain source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._source.point_source_position(reference_position, draw_area)

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self._source.extended_source_magnitude(band=band)

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        return self._source.point_source_magnitude(
            band=band, image_observation_times=image_observation_times
        )

    def kwargs_extended_light(
        self, reference_position=None, draw_area=None, band=None
    ):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param reference_position: reference position. the source position will be
         defined relative to this position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """

        return self._source.kwargs_extended_light(
            reference_position, draw_area, band
        )

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius of a
        galaxy.

        :param band: Imaging band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self._source.surface_brightness_reff(band=band)
