from slsim.Sources.SourceTypes.interpolated_image import Interpolated
import numpy as np
import pytest
from numpy import testing as npt
from astropy import cosmology


class TestInterpolated:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        # Create an image
        # Image Parameters
        size = 61
        center_brightness = 90
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

        self.source_dict = {
            "z": 0.5,
            "image": self.test_image,
            "center_x": size // 2,
            "center_y": size // 2,
            "z_data": 0.1,
            "pixel_width_data": 0.05,
            "phi_G": 0.5,
            "mag_i": 21,
        }
        self.source = Interpolated(cosmo=cosmo, **self.source_dict)

    def test_image_redshift(self):
        assert self.source._image_redshift == 0.1

    def test_image(self):
        assert np.all(self.source._image == self.test_image)

    def test_phi(self):
        assert self.source._phi == 0.5

    def test_extended_source_magnitude(self):
        assert self.source.extended_source_magnitude("i") == 21
        with pytest.raises(ValueError):
            self.source.extended_source_magnitude("g")

    def test_kwargs_extended_source_light(self):
        light_model_list, results = self.source.kwargs_extended_light(band="i")
        _, results2 = self.source.kwargs_extended_light(band=None)

        assert np.all(results[0]["image"] == self.test_image)
        npt.assert_almost_equal(results[0]["scale"], 0.0151, decimal=3)
        assert results[0]["phi_G"] == 0.5
        assert results[0]["magnitude"] == 21
        assert results2[0]["magnitude"] == 1

        assert light_model_list[0] == "INTERPOL"


if __name__ == "__main__":
    pytest.main()
