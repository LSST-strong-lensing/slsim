from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy

class SingleSersic(SourceBase):
    """class to manage source with single sersic light profile"""
    def __init__(self, source_dict):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift, a magnitude in any band, 
         sersic index, angular size in arcsec, and ellipticity.
         eg: {"z": 0.8, "mag_i": 22, "n_sersic": 1, "angular_size": 0.10, 
         "e1": 0.002, "e2": 0.001}. One can provide magnitudes in multiple bands.
        :type source_dict: dict or astropy.table.Table
        """
        super().__init__(source_dict = source_dict)

    @property
    def n_sersic(self):
        """Returns sersic index of the source."""

        return float(self.source_dict["n_sersic"])

    @property
    def angular_size(self):
        """Returns angular size of the source."""

        return float(self.source_dict["angular_size"])

    @property
    def ellipticity(self):
        """Returns ellipticity components of source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return float(self.source_dict["e1"]), float(self.source_dict["e2"])
    
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
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position(
            center_lens=center_lens, draw_area=draw_area
        )
        size_source_arcsec = float(self.angular_size)
        e1_light_source_lenstronomy, e2_light_source_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self.ellipticity[0], e2_slsim=self.ellipticity[1]
            )
        )
        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "R_sersic": size_source_arcsec,
                "n_sersic": float(self.n_sersic),
                "e1": e1_light_source_lenstronomy,
                "e2": e2_light_source_lenstronomy,
                "center_x": center_source[0],
                "center_y": center_source[1],
            }
        ]
        return kwargs_extended_source

    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """
        source_models_list = ["SERSIC_ELLIPSE"]
        return source_models_list
