from slsim.Sources.SourceTypes.point_source import PointSource
from slsim.Sources.SourceTypes.extended_source import ExtendedSource
from slsim.Sources.SourceTypes.point_plus_extended_source import PointPlusExtendedSource

_SUPPORTED_SOURCES = ["point_source", "extended", "point_plus_extended"]


class Source(object):
    """Class to manage an individual source."""

    def __init__(
        self,
        source_dict,
        source_type=None,
        extendedsource_type=None,
        pointsource_type=None,
        cosmo=None,
        pointsource_kwargs={},
        extendedsource_kwargs={},
    ):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, Interpolated classes, SupernovaEvent, and Quasar class.
        :type source_dict: dict or astropy.table.Table .
         eg of a supernova plus host galaxy dict: {"z": 0.8, "mag_i": 22, "n_sersic": 1,
           "angular_size": 0.10, "e1": 0.002, "e2": 0.001, "ra_off": 0.001, "dec_off": 0.005}
        :param source_type: Keyword to specify type of the source. Supported source types are
         'extended', 'point_source', and 'point_plus_extended' supported
        :type source_type: str
        :param extendedsource_type: Keyword to specify type of the extended source.
         Supported extended source types are "single_sersic", "double_sersic", "interpolated".
        :param pointsource_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :param cosmo: astropy.cosmology instance
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
        self.cosmo = cosmo
        self.pointsource_kwargs = pointsource_kwargs
        self.extendedsource_kwargs = extendedsource_kwargs
        self.source_type = source_type
        self.extendedsource_type = extendedsource_type
        self.pointsource_type = pointsource_type
        if self.source_type in ["point_source"]:
            self._single_source = PointSource(
                source_dict=source_dict,
                pointsource_type=self.pointsource_type,
                cosmo=self.cosmo,
                pointsource_kwargs=pointsource_kwargs,
            )
        elif self.source_type in ["extended"]:
            self._single_source = ExtendedSource(
                source_dict=source_dict,
                extendedsource_type=self.extendedsource_type,
                cosmo=cosmo,
                extendedsource_kwargs=extendedsource_kwargs,
            )
        elif self.source_type in ["point_plus_extended"]:
            self._single_source = PointPlusExtendedSource(
                source_dict=source_dict,
                extendedsource_type=self.extendedsource_type,
                pointsource_type=self.pointsource_type,
                cosmo=cosmo,
                pointsource_kwargs=pointsource_kwargs,
                extendedsource_kwargs=extendedsource_kwargs,
            )
        else:
            raise ValueError(
                "source type %s not supported. Chose among %s."
                % (source_type, _SUPPORTED_SOURCES)
            )

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._single_source.redshift

    @property
    def angular_size(self):
        """Returns angular size of the extended source."""

        return self._single_source.angular_size

    @property
    def ellipticity(self):
        """Returns ellipticity components of extended source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return self._single_source.ellipticity

    def extended_source_position(self, reference_position=None, draw_area=None):
        """Extended source position. If a center has already been provided (and
        stored in the source_dict), then it is simply returned. Otherwise, a
        source position is drawn uniformly within the circle of the test area
        centered on the deflector position.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a source
         position. The default choice is None. In this case
         source_dict must contain source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._single_source.extended_source_position(
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

        return self._single_source.point_source_position(reference_position, draw_area)

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self._single_source.extended_source_magnitude(band=band)

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        return self._single_source.point_source_magnitude(
            band=band, image_observation_times=image_observation_times
        )

    def kwargs_extended_source_light(
        self, reference_position=None, draw_area=None, band=None
    ):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param reference_position: reference position. the source postion will be
         defined relative to this position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """

        return self._single_source.kwargs_extended_source_light(
            reference_position, draw_area, band
        )

    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """
        return self._single_source.extended_source_light_model()

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius of a
        galaxy.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self._single_source.surface_brightness_reff(band=band)
