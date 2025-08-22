from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Util.param_util import surface_brightness_reff


class SingleSersic(SourceBase):
    """Class to manage source with single sersic light profile."""

    def __init__(self, angular_size, n_sersic, e1=0, e2=0, **source_dict):
        """

        :param angular_size: half-light radius of Sersic [arcsec]
        :param n_sersic: Sersic index
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param source_dict: dictionary for SourceBase() option (see documentation)
        :type source_dict: dict or astropy.table.Table
        """
        super().__init__(
            model_type="SingleSersic",
            extended_source=True,
            point_source=False,
            **source_dict
        )
        self.name = "GAL"
        self._n_sersic = n_sersic
        self._angular_size = angular_size
        self._e1, self._e2 = e1, e2

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Kewords used are in lenstronomy conventions.

        :param reference_position: reference position. the source postion will be
         defined relative to this position. The default choice is None. In this case
         source_dict must contain source position.
         Eg: np.array([0, 0])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi. The default choice is None. In this case
         source_dict must contain source position.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position
        # convert from slsim to lenstronomy convention.
        e1_light_source_lenstronomy, e2_light_source_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1,
                e2_slsim=self._e2,
            )
        )
        light_models_list = [
            "SERSIC_ELLIPSE",
        ]
        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "R_sersic": self.angular_size,
                "n_sersic": self._n_sersic,
                "e1": e1_light_source_lenstronomy,
                "e2": e2_light_source_lenstronomy,
                "center_x": center_source[0],
                "center_y": center_source[1],
            }
        ]
        return light_models_list, kwargs_extended_source

    def surface_brightness_reff(self, band=None):
        """Calculate average surface brightness within half light radius.

        :param band: Imageing band
        :return: average surface brightness within half light radius
            [mag/arcsec^2]
        """
        # reference_position and draw_area do not matter, they are dummy input here.
        light_model_list, kwargs_source = self.kwargs_extended_light(band=band)
        return surface_brightness_reff(
            angular_size=self.angular_size,
            source_model_list=light_model_list,
            kwargs_extended_source=kwargs_source,
        )
