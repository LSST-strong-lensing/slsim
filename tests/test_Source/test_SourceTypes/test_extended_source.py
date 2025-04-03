from slsim.Sources.SourceTypes.extended_source import ExtendedSource
import numpy as np
import pytest
from numpy import testing as npt
from astropy import cosmology


class TestExtendedSource:
    def setup_method(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict_single_sersic = {"z": 1.0, "mag_i": 21, "n_sersic": 1,
                                         "angular_size": 0.2, "e1": 0.005, "e2": 0.003,
                                         "center_x": 0.034, "center_y": -0.06}
        kwargs_single_sersic = {"extendedsource_type": "single_sersic"}
        self.source = ExtendedSource(source_dict=self.source_dict_single_sersic,
                                      cosmo=cosmo, **kwargs_single_sersic)
        
        self.source_dict_double_sersic = {"z": 0.5, "n_sersic_0":1, "n_sersic_1": 4, "angular_size0": 0.2,
                "angular_size1": 0.15, "e0_1": 0.001, "e0_2": 0.002, "e1_1": 0.001,
                "e1_2": 0.003, "w0":0.4, "w1": 0.6, "mag_i": 23,}
        kwargs_double_sersic = {"extendedsource_type": "double_sersic"}
        self.source_double_sersic = ExtendedSource(source_dict=self.source_dict_double_sersic, 
                                                   cosmo=cosmo, **kwargs_double_sersic)
        
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
        
        self.source_dict_interpolated = {"z": 0.5, "image": self.test_image, "center_x": size // 2,
                "center_y": size // 2, "z_data": 0.1, "pixel_width_data":0.05,
                "phi_G": 0.0, "mag_i": 20,
        }
        kwargs_interpolated = {"extendedsource_type": "interpolated"}
        self.source_interpolated = ExtendedSource(source_dict=self.source_dict_interpolated, 
                                                  cosmo=cosmo, **kwargs_interpolated)
        
    def test_redshift(self):
        assert self.source.redshift == 1.0

    def test_angular_size(self):
        assert self.source.angular_size == 0.2

    def test_ellipticity(self):
        e1, e2 = self.source.ellipticity
        assert e1 == 0.005
        assert e2 == 0.003
    
    def test_n_sersic(self):
        assert self.source.n_sersic == 1

    def test_sersicweight(self):
        w0, w1=self.source_double_sersic.sersicweight
        assert w0==0.4
        assert w1==0.6

    def test_extended_source_position(self):
        x_pos, y_pos = self.source.extended_source_position(reference_postion=[0, 0],
                                                             draw_area=4*np.pi)
        assert x_pos == 0.034
        assert y_pos == -0.06

    def test_extended_source_magnitude(self):
        assert self.source.extended_source_magnitude("i") == 21

    def test_kwargs_extended_source_light(self):
        results = self.source.kwargs_extended_source_light(band="i", reference_position=[0, 0],
                                                             draw_area=4*np.pi)
        assert results[0]["R_sersic"] == 0.2
        assert results[0]["center_x"] == 0.034
        assert results[0]["center_y"] == -0.06
        assert results[0]["e1"] == -0.005
        assert results[0]["e2"] == 0.003
        assert results[0]["magnitude"] == 21

    def test_extended_source_light_model(self):
        source_model = self.source.extended_source_light_model()
        assert source_model[0] == "SERSIC_ELLIPSE"

    def test_surface_brightness_reff(self):
        result = self.source.surface_brightness_reff(band="i")
        npt.assert_almost_equal(result, 19.500, decimal=3)

    def test_image_redshift(self):
        assert self.source_interpolated.image_redshift == 0.1
    
    def test_image(self):
        image_result = self.source_interpolated.image
        assert np.all(image_result == self.test_image)
    
    def test_phi(self):
        phi = self.source_interpolated.phi
        assert phi == 0.0

    def test_pixel_scale(self):
        delta_pix = self.source_interpolated.pixel_scale
        assert delta_pix==0.05
    
    def test_error(self):
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict_extended = {"z": 1.0, "mag_i": 21, "n_sersic": 1,
                                         "angular_size": 0.2, "e1": 0.005, "e2": 0.003,
                                         "center_x": 0.034, "center_y": -0.06}
        kwargs_extended = {"extendedsource_type": "other"}
        with pytest.raises(ValueError):
            ExtendedSource(source_dict=self.source_dict_extended,
                              cosmo=cosmo, **kwargs_extended)

if __name__ == "__main__":
    pytest.main()
