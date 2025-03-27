from slsim.Sources.SourceTypes.single_sersic import SingleSersic
from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Sources.SourceTypes.interpolated_image import Interpolated

_SUPPORTED_SOURCES = ["single_sersic", "double_sersic", "interpolated"]

class ExtendedSource(object):
    """Class to manage a single extended source"""
    def __init__(self, source_dict, cosmo=None, extendedsource_type=None):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, and Interpolated classes.
        :type source_dict: dict or astropy.table.Table
        :param cosmo: astropy.cosmology instance
        """
        if extendedsource_type in ["single_sersic"]:
            self._source = SingleSersic(source_dict=source_dict)
        elif extendedsource_type in ["double_sersic"]:
            self._source = DoubleSersic(source_dict=source_dict)
        elif extendedsource_type in ["interpolated"]:
            self._source = Interpolated(source_dict=source_dict, cosmo=cosmo)
        else:
            raise ValueError(
                "source type %s not supported. Chose among %s."
                % (extendedsource_type, _SUPPORTED_SOURCES)
            )
        
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