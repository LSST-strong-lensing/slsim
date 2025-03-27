import numpy as np
from slsim.Sources.SourceTypes.extended_source_base import ExtendedSourceBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy

class DoubleSersic(ExtendedSourceBase):
    """class to manage source with double sersic light profile"""
    def __init__(self, source_dict):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift, a magnitude in any band, 
         sersic indices, sersic weight, angular sizes in arcsec, ellipticity.
         eg: {"z": 0.8, "mag_i": 22, "n_sersic_0": 1, "n_sersic_1": 4, "w0": 0.1,
         "w1": 0.9, "angular_size0": 0.10, "angular_size1": 0.05, "e0_1": 0.002, 
         "e0_2": 0.001, "e1_1": 0.0, "e1_2": 0.0}. One can provide magnitudes in 
         multiple bands.
        :type source_dict: dict or astropy.table.Table
        """
        super().__init__(source_dict = source_dict)

    @property
    def n_sersic(self):
        """Returns sersic indices of the source for double sersic profile."""

        return (float(self.source_dict["n_sersic_0"]), 
                float(self.source_dict["n_sersic_1"]))
    
    @property
    def sersicweight(self):
        """Returns weight of the sersic components"""

        return self.source_dict["w0"], self.source_dict["w1"]

    @property
    def angular_size(self):
        """Returns angular size of the source for two component of
          the sersic profile."""

        return (float(self.source_dict["angular_size0"]),
                 float(self.source_dict["angular_size1"]))

    @property
    def ellipticity(self):
        """Returns ellipticity components of source for the both component of 
        the light profile. first two ellipticity components are associated with the 
        first sersic component and last two are associated with the second sersic component.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return (float(self.source_dict["e0_1"]), float(self.source_dict["e0_2"]),
                 float(self.source_dict["e1_1"]), float(self.source_dict["e1_2"]))
    
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
        # compute magnitude for each sersic component based on weight
        flux = 10 ** (-mag_source / 2.5)
        mag_source0 = -2.5 * np.log10(self.sersicweight[0] * flux)
        mag_source1 = -2.5 * np.log10(self.sersicweight[1] * flux)
        # convert from slsim to lenstronomy convention.
        e1_light_source_1_lenstronomy, e2_light_source_1_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self.ellipticity[0],
                e2_slsim=self.ellipticity[1],
            )
        )
        e1_light_source_2_lenstronomy, e2_light_source_2_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self.ellipticity[2],
                e2_slsim=self.ellipticity[3],
            )
        )
        kwargs_extended_source = [
            {
                "magnitude": mag_source0,
                "R_sersic": self.angular_size[0],
                "n_sersic": self.n_sersic[0],
                "e1": e1_light_source_1_lenstronomy,
                "e2": e2_light_source_1_lenstronomy,
                "center_x": center_source[0],
                "center_y": center_source[1],
            },
            {
                "magnitude": mag_source1,
                "R_sersic": self.angular_size[1],
                "n_sersic": self.n_sersic[1],
                "e1": e1_light_source_2_lenstronomy,
                "e2": e2_light_source_2_lenstronomy,
                "center_x": center_source[0],
                "center_y": center_source[1],
            },
        ]
        return kwargs_extended_source

    def extended_source_light_model(self):
        """Provides a list of source models.

        :return: list of extented source model.
        """
        source_models_list = [
                "SERSIC_ELLIPSE",
                "SERSIC_ELLIPSE",
            ]
        return source_models_list