from slsim.Sources.source import Source
import numpy as np
import pytest
from numpy import testing as npt
from astropy import cosmology


class TestSource:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict_extended = {
            "z": 1.0,
            "mag_i": 21,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.005,
            "e2": 0.003,
            "center_x": 0.034,
            "center_y": -0.06,
        }
        self.source = Source(
            cosmo=cosmo,
            extended_source_type="single_sersic",
            **self.source_dict_extended,
        )

        self.source_dict_point_extended = {
            "z": 1.0,
            "ps_mag_i": 20,
            "mag_i": 21,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.005,
            "e2": 0.003,
            "center_x": 0.044,
            "center_y": -0.05,
            "ra_off": 0.001,
            "dec_off": 0.002,
        }

        kwargs_point_extended = {
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i", "r"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None,
        }
        self.source_point_extended = Source(
            extended_source_type="single_sersic",
            point_source_type="supernova",
            cosmo=cosmo,
            **self.source_dict_point_extended,
            **kwargs_point_extended,
        )

        # Create an image
        # Image Parameters
        size = 100
        center_brightness = 100
        noise_level = 10

        # Create a grid of coordinates
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)

        # Calculate the distance from the center
        r = np.sqrt(x**2 + y**2)

        # Create the galaxy image with light concentrated near the center
        image = center_brightness * np.exp(-(r**2) / 0.1)

        # Add noise to the image
        noise = noise_level * np.random.normal(size=(size, size))
        image += noise

        # Ensure no negative values
        image = np.clip(image, 0, None)
        self.test_image = image

        self.source_dict_interpolated = {
            "z": 0.5,
            "image": self.test_image,
            "center_x": size // 2,
            "center_y": size // 2,
            "z_data": 0.1,
            "pixel_width_data": 0.05,
            "phi_G": 0.0,
            "mag_i": 20,
        }
        self.source_interpolated = Source(
            cosmo=cosmo,
            extended_source_type="interpolated",
            **self.source_dict_interpolated,
        )

        self.source_dict_point = {
            "z": 1.0,
            "ps_mag_i": 20,
            "center_x": 0.044,
            "center_y": -0.05,
        }
        kwargs_point = {
            "variability_model": "light_curve",
            "kwargs_variability": ["supernovae_lightcurve", "i", "r"],
            "sn_type": "Ia",
            "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab",
            "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None,
        }
        self.source_point = Source(
            point_source_type="supernova",
            cosmo=cosmo,
            **self.source_dict_point,
            **kwargs_point,
        )

        self.source_general_lc = Source(
            point_source_type="general_lightcurve", z=1, MJD=[0, 1, 2]
        )

    def test_redshift(self):
        assert self.source.redshift == 1.0

    def test_angular_size(self):
        assert self.source.angular_size == 0.2

    def test_ellipticity(self):
        e1, e2 = self.source.ellipticity
        assert e1 == 0.005
        assert e2 == 0.003

    def test_extended_source_position(self):
        x_pos, y_pos = self.source.extended_source_position
        assert x_pos == 0.034
        assert y_pos == -0.06

    def test_extended_source_magnitude(self):
        assert self.source.extended_source_magnitude("i") == 21

    def test_kwargs_extended_source_light(self):
        source_model, results = self.source.kwargs_extended_light(band="i")
        assert results[0]["R_sersic"] == 0.2
        assert results[0]["center_x"] == 0.034
        assert results[0]["center_y"] == -0.06
        assert results[0]["e1"] == -0.005
        assert results[0]["e2"] == 0.003
        assert results[0]["magnitude"] == 21

    def test_extended_source_light_model(self):
        source_model, kwargs_light = self.source.kwargs_extended_light()
        assert source_model[0] == "SERSIC_ELLIPSE"

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 19.500, decimal=3)

    def test_point_source_position(self):
        x_pos, y_pos = self.source_point_extended.point_source_position
        assert x_pos == 0.045
        assert y_pos == -0.048

    def test_point_source_magnitude(self):
        result = self.source_point_extended.point_source_magnitude(band="i")
        assert result == 20

    def test_point_source_only(self):
        x_pos_1, y_pos_1 = self.source_point.point_source_position
        x_pos_2, y_pos_2 = self.source_point.point_source_position
        assert x_pos_1 == x_pos_2
        assert y_pos_1 == y_pos_2

    def test_error(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict_extended = {
            "z": 1.0,
            "mag_i": 21,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.005,
            "e2": 0.003,
            "center_x": 0.034,
            "center_y": -0.06,
        }
        with pytest.raises(ValueError):
            Source(
                source_dict=self.source_dict_extended,
                source_type="other",
                cosmo=cosmo,
                extendedsource_type="single_sersic",
            )


if __name__ == "__main__":
    pytest.main()
