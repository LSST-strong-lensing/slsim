from slsim.Sources.SourceTypes.point_source import PointSource
from slsim.Sources.SourceTypes.extended_source import ExtendedSource
from slsim.Sources.SourceTypes.point_plus_extended_source import PointPlusExtendedSource

_SUPPORTED_SOURCES = ["point_source", "extended_source", "point_plus_extended"]

class Source(object):
    """class to manage an individual source."""

    def __init__(
        self,
        source_dict,
        source_type=None,
        cosmo=None,
        **kwargs
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
        :param cosmo: astropy.cosmology instance
        :param kwargs: dictionary of keyword arguments for a source. It should 
         contain keywords for pointsource_type or extendedsource_type and other keywords associated with 
         pointsource. For supernova kwargs dict, please see documentation of 
         SupernovaEvent class. For quasar kwargs dict, please see documentation of 
         Quasar class. For extended source, only extentedsource_type is enough. 
         eg: {"extedndedsource_type": "single_sersic"}
         Eg of supernova kwargs: kwargs={"pointsource_type": "supernova", 
          "variability_model": "light_curve", "kwargs_variability": ["supernovae_lightcurve",
            "i", "r"], "sn_type": "Ia", "sn_absolute_mag_band": "bessellb", 
            "sn_absolute_zpsys": "ab", "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": "/Users/narayankhadka/Downloads/sncosmo_sn_models/SALT3.NIR_WAVEEXT/"}.
         Other supported pointsource_types are "supernova", "quasar".
        """
        self.cosmo = cosmo
        if source_type in ["point_source"]:
            self.__source = PointSource(source_dict=source_dict, cosmo=self.cosmo,
                                         **kwargs)
        elif source_type in ["extended_source"]:
            self.__source = ExtendedSource(source_dict=source_dict, cosmo=cosmo, **kwargs)
        elif source_type in ["point_plus_extended"]:
            self.__source = PointPlusExtendedSource(source_dict=source_dict, cosmo=cosmo,
                                                **kwargs)
        else:
            raise ValueError(
                "source type %s not supported. Chose among %s."
                % (source_type, _SUPPORTED_SOURCES)
            )
        
    @property
    def redshift(self):
        """Returns source redshift."""

        return self.__source.redshift
    
    @property
    def angular_size(self):
        """Returns angular size of the source."""

        return self.__source.angular_size
    
    @property
    def ellipticity(self):
        """Returns ellipticity components of source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """
        
        return self.__source.ellipticity
    
    @property
    def n_sersic(self):
        """Returns sersic indices of the source profile."""

        return self.__source.n_sersic

    @property
    def sersicweight(self):
        """Returns weight of the sersic components"""

        return self._source.sersicweight
    
    @property
    def image_redshift(self):
        """Returns redshift of a given image"""
        
        return self.__source.image_redshift
    
    @property
    def image(self):
        """Returns image of a given extended source"""
        
        return self.__source.image

    @property
    def phi(self):
        """Returns position angle of a given image in arcsec"""

        return self.__source.phi
    
    @property
    def pixel_scale(self):
        """Returns pixel scale of a given image"""

        return self.__source.pixel_scale
    
    @property
    def extended_source_position(self):
        """source position. If a center has already been provided (and
        stored in self._center_source during initialization), then it is simply
        returned. Otherwise, a source position is drawn uniformly within the
        circle of the test area centered on the deflector position.

        :param center_lens: center of the deflector.
            Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a source
            position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """
        
        return self.__source.extended_source_position
    
    @property
    def point_source_position(self):
        """Point source position. point source could be at the center of the
        extended source or it can be off from center of the extended source.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self.__source.point_source_position
        
    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self.__source.extended_source_magnitude(band=band)
    
    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        return self.__source.point_source_magnitude(band=band, 
                    image_observation_times=image_observation_times)
    
    def kwargs_extended_source_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        
        return self.__source.kwargs_extended_source_light(band=band)
    
    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """
        return self.__source.extended_source_light_model()
    
    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius of a galaxy.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self.__source.surface_brightness_reff(band=band)