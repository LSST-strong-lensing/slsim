from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util.cosmo_util import z_scale_factor


class Interpolated(SourceBase):
    """Class to manage source with real extended source image."""

    def __init__(self, image, pixel_width_data, phi_G, z_data, cosmo, **source_dict):
        """
        :param image: pixelated image to be interpolated
        :type image: 2d numpy array
        :param pixel_width_data: width of pixel in image data [arcseconds], which is then being redshifted to the
         redshift of the source
        :param z_data: redshift of the original image data
        :param phi_G: rotation angle of the interpolated image in regard to the RA-DEC coordinate grid
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
        super().__init__(
            extended_source=True, point_source=False, cosmo=cosmo, **source_dict
        )
        self.name = "GAL"
        self._image = image
        self._pixel_scale = pixel_width_data
        self._z_data = z_data
        self._phi = phi_G

    @property
    def _image_redshift(self):
        """Returns redshift of a given image."""

        return self._z_data

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position
        pixel_width = self._pixel_scale
        pixel_width *= z_scale_factor(
            z_old=self._image_redshift, z_new=self.redshift, cosmo=self._cosmo
        )
        light_model_list = ["INTERPOL"]
        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "image": self._image,  # Use the potentially reshaped image
                "center_x": center_source[0],
                "center_y": center_source[1],
                "phi_G": self._phi,
                "scale": pixel_width,
            }
        ]
        return light_model_list, kwargs_extended_source
