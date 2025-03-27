from slsim.Sources.SourceTypes.single_sersic import SingleSersic
from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Sources.SourceTypes.interpolated_image import Interpolated

_SUPPORTED_SOURCES = ["single_sersic", "double_sersic", "interpolated"]

class ExtendedSource(object):
    """Class to manage a single extended source"""
    def __init__(self, source_dict, extendedsource_type="single_sersic", cosmo=None):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, and Interpolated classes.
        :type source_dict: dict or astropy.table.Table
        :param extendedsource_type: keyword for specifying light profile model.
        :type extendedsource_type: str. supported types are "single_sersic", 
         "double_sersic", "interpolated"
        :param cosmo: astropy.cosmology instance
        """
        self._extendedsource_type = extendedsource_type
        if self._extendedsource_type in ["single_sersic"]:
            self._source = SingleSersic(source_dict=source_dict)
        elif self._extendedsource_type in ["double_sersic"]:
            self._source = DoubleSersic(source_dict=source_dict)
        elif self._extendedsource_type in ["interpolated"]:
            self._source = Interpolated(source_dict=source_dict, cosmo=cosmo)
        else:
            raise ValueError(
                "source type %s not supported. Chose among %s."
                % (self._extendedsource_type, _SUPPORTED_SOURCES)
            )
    
    @property
    def redshift(self):
        """Returns source redshift."""

        return self._source.redshift
    
    @property
    def angular_size(self):
        """Returns angular size of the source."""

        return self._source.angular_size
    
    @property
    def ellipticity(self):
        """Returns ellipticity components of source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """
        
        return self._source.ellipticity
    
    @property
    def n_sersic(self):
        """Returns sersic indices of the source profile."""

        return self._source.n_sersic

    @property
    def sersicweight(self):
        """Returns weight of the sersic components"""

        return self._source.sersicweight
    
    @property
    def image_redshift(self):
        """Returns redshift of a given image"""
        
        return self._source.image_redshift
    
    @property
    def image(self):
        """Returns image of a given extended source"""
        
        return self._source.image

    @property
    def phi(self):
        """Returns position angle of a given image in arcsec"""

        return self._source.phi
    
    @property
    def pixel_scale(self):
        """Returns pixel scale of a given image"""

        return self._source.pixel_scale
        
    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self._source.extended_source_magnitude(band=band)
        
    def extended_source_position(self, center_lens, draw_area):
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
        
        return self._source.extended_source_position(center_lens=center_lens,
                                                 draw_area=draw_area)
    
    def kwargs_extended_source_light(self, center_lens, draw_area, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        
        return self._source.kwargs_extended_source_light(center_lens=center_lens,
                                                 draw_area=draw_area, band=band)
    
    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """
        return self._source.extended_source_light_model()