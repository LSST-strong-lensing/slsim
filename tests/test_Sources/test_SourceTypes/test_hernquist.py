from slsim.Sources.SourceTypes.hernquist import Hernquist
import numpy.testing as npt


class TestHernquist(object):

    def setup_method(self):

        self.model = Hernquist(
            z=1, angular_size=1.0, e1=0.1, e2=-0.1, stellar_mass=1e11, mag_g=10
        )

    def test_angular_size(self):
        angular_size = self.model.angular_size
        assert angular_size == 1.0

    def test_kwargs_extended_light(self):
        light_model, kwargs_light = self.model.kwargs_extended_light(band="g")
        assert light_model[0] == "HERNQUIST_ELLIPSE"
        assert kwargs_light[0]["Rs"] == 1.0 / 1.815

    def test_surface_brightness_reff(self):
        surf_bright = self.model.surface_brightness_reff(band="g")
        npt.assert_almost_equal(surf_bright, 11.995449, decimal=5)
