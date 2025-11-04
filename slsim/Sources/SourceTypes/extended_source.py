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
            self._extended_source = SingleSersic(**source_dict)
        elif source_type in ["double_sersic"]:
            self._extended_source = DoubleSersic(**source_dict)
        elif source_type in ["catalog_source"]:
            self._extended_source = CatalogSource(**source_dict)
        elif source_type in ["interpolated"]:
            self._extended_source = Interpolated(**source_dict)
        else:
            raise ValueError(
                "Extended source type %s not supported. Chose among %s."
                % (source_type, _SUPPORTED_EXTENDED_SOURCES)
            )

    @property
    def name(self):
        """Meaningful name string of the source.

        :return: name string
        """
        return self._extended_source.name

    @property
    def redshift(self):
        """Returns source redshift."""

        return self._extended_source.redshift

    def update_center(
        self, area=None, reference_position=None, center_x=None, center_y=None
    ):
        """Overwrites the source center position.

        :param reference_position: [RA, DEC] in arc-seconds of the
            reference from where within a circle the source position is
            being drawn from
        :type reference_position: 2d numpy array
        :param area: area (in solid angle arcseconds^2) to dither the
            center of the source
        :param center_x: RA position [arc-secons] (optional, otherwise
            renders within area)
        :param center_y: DEC position [arc-secons] (optional, otherwise
            renders within area)
        :return: Source() instance updated with new center position
        """
        return self._extended_source.update_center(
            area, reference_position, center_x=center_x, center_y=center_y
        )

    @property
    def angular_size(self):
        """Returns angular size of the source."""

        return self._extended_source.angular_size

    @property
    def ellipticity(self):
        """Returns ellipticity components of source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return self._extended_source.ellipticity

    @property
    def extended_source_position(self):
        """Extended source position.

        :return: [x_pos, y_pos]
        """

        return self._extended_source.extended_source_position

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """

        return self._extended_source.extended_source_magnitude(band=band)

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """

        return self._extended_source.kwargs_extended_light(band=band)

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """

        return self._extended_source.surface_brightness_reff(band=band)
