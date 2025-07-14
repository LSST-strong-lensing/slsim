from slsim.Sources.SourceTypes.supernova_event import SupernovaEvent
from slsim.Sources.SourceTypes.quasar import Quasar
from slsim.Sources.SourceTypes.general_lightcurve import GeneralLightCurve

_SUPPORTED_POINT_SOURCES = ["supernova", "quasar", "general_lightcurve"]


class PointSource(object):
    """Class to manage a single point source."""

    def __init__(self, source_dict, pointsource_type, cosmo=None, pointsource_kwargs={}):
        """One can supply either supernovae kwargs or agn kwargs. If supernovae
        kwargs are supplied, agn kwrgs can be None which is a default option.

        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For more detail, please see documentation of SupernovaEvent and Quasar class.
        :type source_dict: dict or astropy.table.Table
        :param pointsource_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :type source_type: str
        :param cosmo: astropy.cosmology instance
        :param pointsource_kwargs: dictionary of keyword arguments for a point source. It should
         contain keywords for pointsource_type and other keywords associated with
         supernova and quasar. For supernova kwargs dict, please see documentation of
         SupernovaEvent class. For quasar kwargs dict, please see documentation of
         Quasar class.
         Eg of supernova kwargs: kwargs={"variability_model": "light_curve",
         "kwargs_variability": ["supernovae_lightcurve", "i", "r"], "sn_type": "Ia",
         "sn_absolute_mag_band": "bessellb", "sn_absolute_zpsys": "ab",
         "lightcurve_time": np.linspace(-50, 100, 150),
         "sn_modeldir": "/Users/narayankhadka/Downloads/sncosmo_sn_models/SALT3.NIR_WAVEEXT/"}.
         Other supported pointsource_types are "supernova", "quasar".
        """

        if pointsource_type in ["supernova"]:
            self._point_source = SupernovaEvent(
                source_dict=source_dict, cosmo=cosmo, **pointsource_kwargs
            )
        elif pointsource_type in ["quasar"]:
            self._point_source = Quasar(source_dict=source_dict, cosmo=cosmo, **pointsource_kwargs)
        elif pointsource_type in ["general_lightcurve"]:
            self._point_source = GeneralLightCurve(source_dict=source_dict, **pointsource_kwargs)
        else:
            raise ValueError(
                "Point source type %s not supported. Chose among %s."
                % (pointsource_type, _SUPPORTED_POINT_SOURCES)
            )

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._point_source.redshift

    def extended_source_position(self, reference_position=None, draw_area=None):
        """Provides extended source position if host galaxy is given. In the
        absence of host galaxy, this is the same position as point source
        position. This position is not necessary in this class but we inherite
        this class in PointPlusExtendedSource class where this position is
        necessary.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case
         source_dict must contain source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._point_source.extended_source_position(
            reference_position, draw_area
        )

    def point_source_position(self, reference_position=None, draw_area=None):
        """Point source position. point source could be at the center of the
        extended source or it can be off from center of the extended source.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case
         source_dict must contain source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._point_source.point_source_position(reference_position, draw_area)

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
