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
        pointsource_type=None,
        extendedsource_type=None,
        cosmo=None,
        **kwargs
    ):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
        :type source_dict: dict or astropy.table.Table
        When source_type is 'interpolated', include keys:
        - 'z' (float)
        - 'image' (numpy.ndarray)
        - 'z_data' (float)
        - 'pixel_width_data' (float)
        - 'phi_G' (float)
        - 'center_x' (float)
        - 'center_y' (float)
        :param variability_model: keyword for variability model to be used. This is an
         input for the Variability class.
        :type variability_model: str
        :param kwargs_variability: Keyword arguments for variability class.
         This is associated with an input for Variability class. By using these key
         words, code search for quantities in source_dict with these names and creates
         a dictionary and this dict should be passed to the Variability class.
        :type kwargs_variability: list of str
        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param sn_absolute_mag_band: Band used to normalize to absolute magnitude
        :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
        :param sn_absolute_zpsys: Optional, AB or Vega (AB default)
        :type sn_absolute_zpsys: str
        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array
        :param sn_modeldir: sn_modeldir is the path to the directory containing files
         needed to initialize the sncosmo.model class. For example,
         sn_modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can
         be downloaded from https://github.com/LSST-strong-lensing/data_public .
         For more detail, please look at the documentation of RandomizedSupernovae
         class.
        :type sn_modeldir: str
        :param agn_known_band: Speclite filter of which the magnitude is known. Used to normalize
         mean magnitudes.
        :type agn_known_band: str
        :param agn_known_mag: Magnitude of the agn in the known band.
        :type agn_known_mag: float
        :param agn_driving_variability_model: Variability model with light_curve output
         which drives the variability across all bands of the agn.
        :type agn_driving_variability_model: str (e.g. "light_curve", "sinusoidal", "bending_power_law")
        :param agn_driving_kwargs_variability: Dictionary containing all variability parameters
         for the driving variability class
        :type agn_driving_kwargs_variability: dict
        :param source_type: type of the source 'extended' or 'point_source' or
         'point_plus_extended' supported
        :type source_type: str
        :param light_profile: keyword for number of sersic profile to use in source
         light model
        :type light_profile: str . Either "single_sersic", "double_sersic", or "interpolated" .
        """
        self.cosmo = cosmo
        if source_type in ["point_source"]:
            self.__source = PointSource(source_dict=source_dict, cosmo=self.cosmo,
                                 pointsource_type=pointsource_type, **kwargs)
        elif source_type in ["extended_source"]:
            self.__source = ExtendedSource(source_dict=source_dict, cosmo=cosmo,
                                           extendedsource_type=extendedsource_type)
        elif source_type in ["point_plus_extended"]:
            self.__source = PointPlusExtendedSource(source_dict=source_dict, cosmo=cosmo,
                                                pointsource_type=pointsource_type, 
                                                extendedsource_type=extendedsource_type,
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
        """Calculate average surface brightness within half light radius.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self.__source.surface_brightness_reff(band=band)

    