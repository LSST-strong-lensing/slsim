from slsim.Sources.SourceTypes.single_sersic import SingleSersic
from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Sources.SourceTypes.catalog_source import CatalogSource
from slsim.Sources.SourceTypes.interpolated_image import Interpolated

_SUPPORTED_EXTENDED_SOURCES = [
    "single_sersic",
    "double_sersic",
    "catalog_source",
    "interpolated",
]


class ExtendedSource(object):
    """Class to manage a single extended source."""

    def __init__(self, source_type, **source_dict):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For a detailed description of this dictionary, please see the documentation for
         the SingleSersic, DoubleSersic, and Interpolated classes.
        :type source_dict: dict or astropy.table.Table
        :param source_type: Keyword to specify type of the extended source.
         Supported extended source types are "single_sersic", "double_sersic", "catalog_source", "interpolated".
        :type source_type: str
        :param cosmo: astropy.cosmology instance
        :param extendedsource_kwargs: dictionary of keyword arguments for specific extended source classes.
         Currently only used for COSMOSSource, see its documentation.
        """
        if source_type in ["single_sersic"]:
            self._source = SingleSersic(**source_dict)
        elif source_type in ["double_sersic"]:
            self._source = DoubleSersic(**source_dict)
        elif source_type in ["catalog_source"]:
            self._source = CatalogSource(**source_dict
            )
        elif source_type in ["interpolated"]:
            self._source = Interpolated(**source_dict)
        else:
            raise ValueError(
                "Extended source type %s not supported. Chose among %s."
                % (source_type, _SUPPORTED_EXTENDED_SOURCES)
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

    def extended_source_position(self, reference_postion=None, draw_area=None):
        """Extended source position. If a center has already been provided (and
        stored in self._center_source during initialization of _source), then
        it is simply returned. Otherwise, a source position is drawn uniformly
        within the circle of the test area centered on the deflector position.
        see: _source.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a source
         position. The default choice is None. In this case source_dict must contain
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._source.extended_source_position(reference_postion, draw_area)

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self._source.extended_source_magnitude(band=band)

    def kwargs_extended_light(
        self, reference_position=None, draw_area=None, band=None
    ):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case
         source_dict must contain source position. Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """

        return self._source.kwargs_extended_light(
            reference_position, draw_area, band
        )

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self._source.surface_brightness_reff(band=band)
