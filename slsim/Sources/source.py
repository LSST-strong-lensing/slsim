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
        :param extended_source_type: Keyword to specify type of the extended source. Supported
         extended source types are 'single_sersic', 'double_sersic', 'catalog_source', and 'interpolated'.
        :type extended_source_type: str or None
        :param point_source_type: Keyword to specify type of point source. Supported point
         source types are 'supernova', 'quasar', and 'general_lightcurve'.
        :type point_source_type: str or None
        :param source_dict: Source properties. Can be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the individual classes, such as SingleSersic, DoubleSersic, Interpolated classes, SupernovaEvent,
         and Quasar class.
        :type source_dict: dict or astropy.table.Table .

        """
        self.extended_source_type = extended_source_type
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
            raise ValueError(
                "either extended_source_type or point_source_type needs to be set."
            )

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
                **source_dict,
            )
        else:
            raise ValueError(
                "source type %s not supported. Chose among %s for extended sources and %s for point sources."
                % (source_type, _SUPPORTED_EXTENDED_SOURCES, _SUPPORTED_POINT_SOURCES)
            )

    @property
    def name(self):
        """Meaningful name string of the source.

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

    def update_center(
        self, area=None, reference_position=None, center_x=None, center_y=None
    ):
        """Overwrites the source center position.

        :param reference_position: [RA, DEC] in arc-seconds of the
            reference from where within a circle the source position is
            being drawn from
        :type reference_position: 2d numpy array
        :param area: area (in solid angle arc-seconds^2) to dither the
            center of the source
        :param center_x: RA position [arc-seconds] (optional, otherwise
            renders within area)
        :param center_y: DEC position [arc-seconds] (optional, otherwise
            renders within area)
        :return: Source() instance updated with new center position
        """
        self._source.update_center(
            area=area,
            reference_position=reference_position,
            center_x=center_x,
            center_y=center_y,
        )

    @property
    def extended_source_position(self):
        """Extended source position.

        :return: [x_pos, y_pos]
        """

        return self._source.extended_source_position

    @property
    def point_source_position(self):
        """Point source position. point source could be at the center of the
        extended source, or it can be off from center of the extended source.

        :return: [x_pos, y_pos]
        """

        return self._source.point_source_position

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

    def point_source_type(self, image_positions=False):
        """Type of point source model.

        :param image_positions:
        :return: point source model string, or None
        """
        return self._source.point_source_type(image_positions=image_positions)

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """

        return self._source.kwargs_extended_light(band=band)

    def kwargs_point_source(
        self,
        band,
        image_observation_times=None,
        image_pos_x=None,
        image_pos_y=None,
        ps_mag=None,
    ):
        """

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an image.
        :param image_pos_x: pre-calculated image positions (solutions of the lens equation) RA [arcseconds]
        :param image_pos_y: pre-calculated image positions (solutions of the lens equation) DEC [arcseconds]
        :param ps_mag: magnitudes of images (or source)
        :return: source type, list of dictionary in lenstronomy convention
        """
        return self._source.kwargs_point_source(
            band=band,
            image_observation_times=image_observation_times,
            image_pos_x=image_pos_x,
            image_pos_y=image_pos_y,
            ps_mag=ps_mag,
        )

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius of a
        galaxy.

        :param band: Imaging band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self._source.surface_brightness_reff(band=band)
