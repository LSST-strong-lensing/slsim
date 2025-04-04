from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util.cosmo_util import z_scale_factor


class Interpolated(SourceBase):
    """Class to manage source with real extended source image."""

    def __init__(self, source_dict, cosmo):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift of a source, real image
         associated with the source, redshift of that image, orientation angle of th
         a magnitude in any band, image, redshift of the image, position angle of the
         given image, pixel scale of the image.
         eg: {"z": [0.8], "mag_i": [22], "image": [np.array([[1,2,3], [3,2,4], [5, 2,1]])],
         "z_data": [1.2], "phi_G": [0.1], "pixel_width_data": [0.05]}. One can also add
         magnitudes in multiple bands.
        :type source_dict: dict or astropy.table.Table
        :param cosmo: astropy.cosmology instance
        """
        super().__init__(source_dict=source_dict)
        self.cosmo = cosmo

    @property
    def image_redshift(self):
        """Returns redshift of a given image."""

        return float(self.source_dict["z_data"])

    @property
    def image(self):
        """Returns image of a given extended source."""

        return self.source_dict["image"]

    @property
    def phi(self):
        """Returns position angle of a given image in arcsec."""

        return self.source_dict["phi_G"]

    @property
    def pixel_scale(self):
        """Returns pixel scale of a given image."""

        return self.source_dict["pixel_width_data"]

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """
        column_names = self.source_dict.colnames
        if "mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "mag_" + band
        source_mag = self.source_dict[band_string]
        return source_mag

    def kwargs_extended_source_light(self, reference_position, draw_area, band=None):
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
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position(
            reference_position=reference_position, draw_area=draw_area
        )
        z_image = self.image_redshift
        pixel_width = self.pixel_scale
        pixel_width *= z_scale_factor(
            z_old=z_image, z_new=self.redshift, cosmo=self.cosmo
        )

        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "image": self.image,  # Use the potentially reshaped image
                "center_x": center_source[0],
                "center_y": center_source[1],
                "phi_G": self.phi,
                "scale": pixel_width,
            }
        ]
        return kwargs_extended_source

    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """

        source_models_list = ["INTERPOL"]
        return source_models_list
