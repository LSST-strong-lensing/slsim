from slsim.Sources.SourceTypes.extended_source import ExtendedSource
import numpy as np
import pytest
import os
import pathlib

from numpy import testing as npt
from astropy import cosmology

catalog_path = os.path.join(
    str(pathlib.Path(__file__).parent.parent.parent.parent),
    "data",
    "test_COSMOS_23.5_training_sample",
)


class TestExtendedSource:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict_single_sersic = {
            "z": 1.0,
            "mag_i": 21,
            "n_sersic": 1,
            "angular_size": 0.2,
            "e1": 0.005,
            "e2": 0.003,
            "center_x": 0.034,
            "center_y": -0.06,
        }
        self.source = ExtendedSource(
            cosmo=cosmo, source_type="single_sersic", **self.source_dict_single_sersic
        )

        self.source_dict_double_sersic = {
            "z": 0.5,
            "n_sersic_0": 1,
            "n_sersic_1": 4,
            "angular_size_0": 0.2,
            "angular_size_1": 0.15,
            "e0_1": 0.001,
            "e0_2": 0.002,
            "e1_1": 0.001,
            "e1_2": 0.003,
            "w0": 0.4,
            "w1": 0.6,
            "mag_i": 23,
        }

        self.source_double_sersic = ExtendedSource(
            cosmo=cosmo, source_type="double_sersic", **self.source_dict_double_sersic
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

        self.source_interpolated = ExtendedSource(
            cosmo=cosmo, source_type="interpolated", **self.source_dict_interpolated
        )

        self.catalogue_source = ExtendedSource(
            cosmo=cosmo,
            source_type="catalog_source",
            catalog_type="HST_COSMOS",
            catalog_path=catalog_path,
            **self.source_dict_single_sersic
        )

    def test_redshift(self):
        assert self.source.redshift == 1.0

    def test_angular_size(self):
        assert self.source.angular_size == 0.2

    def test_name(self):
        assert self.source.name == "GAL"

    def test_update_center(self):
        self.source.update_center(center_x=1, center_y=1)
        assert self.source.extended_source_position[0] == 1

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
        source_model, results = self.source.kwargs_extended_light(
            band="i",
        )
        assert results[0]["R_sersic"] == 0.2
        assert results[0]["center_x"] == 0.034
        assert results[0]["center_y"] == -0.06
        assert results[0]["e1"] == -0.005
        assert results[0]["e2"] == 0.003
        assert results[0]["magnitude"] == 21

        assert source_model[0] == "SERSIC_ELLIPSE"

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 19.500, decimal=3)

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
            ExtendedSource(
                source_dict=self.source_dict_extended,
                cosmo=cosmo,
                source_type="other",
            )


if __name__ == "__main__":
    pytest.main()
