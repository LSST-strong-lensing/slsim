import numpy as np
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Util.param_util import surface_brightness_reff


class DoubleSersic(SourceBase):
    """Class to manage source with double sersic light profile."""

    def __init__(
        self,
        angular_size_0,
        angular_size_1,
        n_sersic_0,
        n_sersic_1,
        w0,
        w1=None,
        e1_1=0,
        e2_1=0,
        e1_2=0,
        e2_2=0,
        **source_dict
    ):
        """

        :param angular_size_0: half-light radius of the first Sersic component [arcsec]
        :param angular_size_1: half-light radius of the second Sersic component [arcsec]
        :param n_sersic_0: Sersic index of first Sersic component
        :param n_sersic_1: Sersic index of first Sersic component
        :param e1_1: eccentricity component of first Sersic
        :param e2_1: eccentricity component of first Sersic
        :param e1_2: eccentricity component of second Sersic
        :param e2_2: eccentricity component of second Sersic
        :param w0: flux weight of first Sersic component
        :param w1: flux weight of second Sersic component, if =None, will be set w1 = 1 - w0, otherwise it has to match.

        :param source_dict: dictionary for SourceBase() option (see documentation)
        :type source_dict: dict or astropy.table.Table
        """
        super().__init__(
            model_type="DoubleSersic",
            extended_source=True,
            point_source=False,
            **source_dict
        )
        self.name = "GAL"
        self._n_sersic = [n_sersic_0, n_sersic_1]
        self._angular_size_list = [angular_size_0, angular_size_1]
        self._e1_1, self._e2_1 = e1_1, e2_1
        self._e1_2, self._e2_2 = e1_2, e2_2

        s = w0 + w1
        w0 = w0 / s
        self._w0 = w0
        w1 = 1 - w0
        assert np.isclose(w0 + w1, 1, rtol=1e-3)
        self._w1 = w1

        self._light_model_list = [
            "SERSIC_ELLIPSE",
            "SERSIC_ELLIPSE",
        ]

    @property
    def angular_size(self):
        """Returns angular size of the source for two component of the sersic
        profile."""
        if self._angular_size is None:
            from lenstronomy.Analysis.light_profile import LightProfileAnalysis
            from lenstronomy.LightModel.light_model import LightModel

            kwargs_light = self._shape_light_model()
            light_model = LightModel(light_model_list=self._light_model_list)
            analysis = LightProfileAnalysis(light_model=light_model)
            r_eff = analysis.half_light_radius(
                kwargs_light=kwargs_light,
                grid_num=500,
                grid_spacing=np.max(self._angular_size_list) / 10,
            )
            self._angular_size = r_eff

        return self._angular_size

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
        if self._e1 is None or self._e2 is None:
            from lenstronomy.Analysis.light_profile import LightProfileAnalysis
            from lenstronomy.LightModel.light_model import LightModel

            kwargs_light = self._shape_light_model()
            light_model = LightModel(light_model_list=self._light_model_list)
            analysis = LightProfileAnalysis(light_model=light_model)
            self._e1, self._e2 = analysis.ellipticity(
                kwargs_light=kwargs_light,
                grid_num=500,
                grid_spacing=np.max(self._angular_size_list) / 10,
            )

        return self._e1, self._e2

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position
        # compute magnitude for each sersic component based on weight
        flux = 10 ** (-mag_source / 2.5)
        mag_source0 = -2.5 * np.log10(self._w0 * flux)
        mag_source1 = -2.5 * np.log10(self._w1 * flux)
        # convert from slsim to lenstronomy convention.
        e1_light_source_1_lenstronomy, e2_light_source_1_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1_1,
                e2_slsim=self._e2_1,
            )
        )
        e1_light_source_2_lenstronomy, e2_light_source_2_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1_2,
                e2_slsim=self._e2_2,
            )
        )

        kwargs_extended_source = [
            {
                "magnitude": mag_source0,
                "R_sersic": self._angular_size_list[0],
                "n_sersic": self._n_sersic[0],
                "e1": e1_light_source_1_lenstronomy,
                "e2": e2_light_source_1_lenstronomy,
                "center_x": center_source[0],
                "center_y": center_source[1],
            },
            {
                "magnitude": mag_source1,
                "R_sersic": self._angular_size_list[1],
                "n_sersic": self._n_sersic[1],
                "e1": e1_light_source_2_lenstronomy,
                "e2": e2_light_source_2_lenstronomy,
                "center_x": center_source[0],
                "center_y": center_source[1],
            },
        ]
        return self._light_model_list, kwargs_extended_source

    def _shape_light_model(self):
        """

        :return: kwargs_light with correct shape centered at (0,0) with amplitude instead of magnitude
        """
        e1_light_source_1_lenstronomy, e2_light_source_1_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1_1,
                e2_slsim=self._e2_1,
            )
        )
        e1_light_source_2_lenstronomy, e2_light_source_2_lenstronomy = (
            ellipticity_slsim_to_lenstronomy(
                e1_slsim=self._e1_2,
                e2_slsim=self._e2_2,
            )
        )

        kwargs_extended_source = [
            {
                "amp": self._w0,
                "R_sersic": self._angular_size_list[0],
                "n_sersic": self._n_sersic[0],
                "e1": e1_light_source_1_lenstronomy,
                "e2": e2_light_source_1_lenstronomy,
                "center_x": 0,
                "center_y": 0,
            },
            {
                "amp": self._w1,
                "R_sersic": self._angular_size_list[1],
                "n_sersic": self._n_sersic[1],
                "e1": e1_light_source_2_lenstronomy,
                "e2": e2_light_source_2_lenstronomy,
                "center_x": 0,
                "center_y": 0,
            },
        ]
        return kwargs_extended_source

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
