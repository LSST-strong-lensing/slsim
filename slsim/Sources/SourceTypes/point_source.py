from slsim.Sources.SourceTypes.supernova_event import SupernovaEvent
from slsim.Sources.SourceTypes.quasar import Quasar
from slsim.Sources.SourceTypes.general_lightcurve import GeneralLightCurve

_SUPPORTED_POINT_SOURCES = ["supernova", "quasar", "general_lightcurve"]


class PointSource(object):
    """Class to manage a single point source."""

    def __init__(self, source_type, **source_dict):
        """One can supply either supernovae kwargs or agn kwargs. If supernovae
        kwargs are supplied, agn kwargs can be None which is a default option.

        :param source_dict: Source properties. Can be a dictionary or an
            Astropy table. For more detail, please see documentation of
            SupernovaEvent and Quasar class.
        :type source_dict: dict or astropy.table.Table
        """
        if source_type in ["supernova"]:
            self._point_source = SupernovaEvent(**source_dict)
        elif source_type in ["quasar"]:
            self._point_source = Quasar(**source_dict)
        elif source_type in ["general_lightcurve"]:
            self._point_source = GeneralLightCurve(**source_dict)
        else:
            raise ValueError(
                "Point source type %s not supported. Chose among %s."
                % (source_type, _SUPPORTED_POINT_SOURCES)
            )

    @property
    def name(self):
        """Meaningful name string of the source.

        :return: name string
        """
        return self._point_source.name

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._point_source.redshift

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
        return self._point_source.update_center(
            area, reference_position, center_x=center_x, center_y=center_y
        )

    @property
    def point_source_position(self):
        """Point source position. point source could be at the center of the
        extended source, or it can be off from center of the extended source.

        :return: [x_pos, y_pos]
        """

        return self._point_source.point_source_position

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        return self._point_source.point_source_magnitude(
            band=band, image_observation_times=image_observation_times
        )

    def point_source_type(self, image_positions=False):
        """Type of point source model.

        :param image_positions:
        :return: point source model string, or None
        """
        return self._point_source.point_source_type(image_positions=image_positions)

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
        return self._point_source.kwargs_point_source(
            band=band,
            image_observation_times=image_observation_times,
            image_pos_x=image_pos_x,
            image_pos_y=image_pos_y,
            ps_mag=ps_mag,
        )
