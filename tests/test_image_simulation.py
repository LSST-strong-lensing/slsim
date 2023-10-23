import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from slsim.lens import Lens
from slsim.lens_pop import LensPop
from astropy.units import Quantity
from slsim.image_simulation import (
    simulate_image,
    sharp_image,
    sharp_rgb_image,
    rgb_image_from_image_list,
    point_source_coordinate_properties,
    image_data_class,
    centered_coordinate_system,
    point_source_image_with_variability,
    point_source_image_without_variability,
    point_source_image_at_time,
)
import pytest


class TestImageSimulation(object):
    def setup_method(self):
        # path = os.path.dirname(slsim.__file__)

        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        print(path, module_path)
        blue_one = Table.read(
            os.path.join(path, "TestData/blue_one_modified.fits"), format="fits"
        )
        red_one = Table.read(
            os.path.join(path, "TestData/red_one_modified.fits"), format="fits"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.source_dict = blue_one
        self.deflector_dict = red_one
        while True:
            gg_lens = Lens(
                source_dict=self.source_dict,
                deflector_dict=self.deflector_dict,
                cosmo=cosmo,
            )
            if gg_lens.validity_test():
                self.gg_lens = gg_lens
                break

    def test_simulate_image(self):
        image = simulate_image(
            lens_class=self.gg_lens,
            band="g",
            num_pix=100,
            add_noise=True,
            observatory="LSST",
        )
        assert len(image) == 100

    def test_sharp_image(self):
        image = sharp_image(
            lens_class=self.gg_lens,
            band="g",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        assert len(image) == 100

    def test_sharp_rgb_image(self):
        image = sharp_rgb_image(
            lens_class=self.gg_lens,
            rgb_band_list=["r", "g", "i"],
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
        )
        assert len(image) == 100

    def test_rgb_image_from_image_list(self):
        image_g = sharp_image(
            lens_class=self.gg_lens,
            band="g",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        image_r = sharp_image(
            lens_class=self.gg_lens,
            band="r",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        image_b = sharp_image(
            lens_class=self.gg_lens,
            band="i",
            mag_zero_point=30,
            delta_pix=0.05,
            num_pix=100,
            with_deflector=True,
        )
        image_list = [image_r, image_g, image_b]
        image = rgb_image_from_image_list(image_list, 0.5)
        assert len(image) == 100


@pytest.fixture
def quasar_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")
    kwargs_variability = {"amp": 1.0, "freq": 0.5}
    return LensPop(
        deflector_type="all-galaxies",
        source_type="quasars",
        variability_model="sinusoidal",
        kwargs_variability=kwargs_variability,
        kwargs_deflector_cut=None,
        kwargs_source_cut=None,
        kwargs_quasars={
            "num_quasars": 50000,
            "z_min": 0.1,
            "z_max": 5,
            "m_min": 17,
            "m_max": 23,
        },
        kwargs_mass2light=None,
        skypy_config=None,
        sky_area=sky_area,
        cosmo=cosmo,
    )


def test_centered_coordinate_system():
    transform_matrix = np.array([[0.2, 0], [0, 0.2]])
    grid = centered_coordinate_system(101, transform_pix2angle=transform_matrix)

    assert grid["ra_at_xy_0"] == -10
    assert grid["dec_at_xy_0"] == -10
    assert np.shape(grid["transform_pix2angle"]) == np.shape(transform_matrix)


def test_image_data_class(quasar_lens_pop_instance):
    trans_matrix_1 = np.array([[0.2, 0], [0, 0.2]])
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    data_class = image_data_class(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        transform_pix2angle=trans_matrix_1,
    )
    results = data_class._x_at_radec_0
    assert results == 50


def test_point_source_image_properties(quasar_lens_pop_instance):
    transform_matrix = np.array([[0.2, 0], [0, 0.2]])
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    result = point_source_coordinate_properties(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        transform_pix2angle=transform_matrix,
    )

    expected_columns = ["deflector_pix", "image_pix", "ra_image", "dec_image"]
    assert result.colnames[0] == expected_columns[0]
    assert result.colnames[1] == expected_columns[1]
    assert result.colnames[2] == expected_columns[2]
    assert result.colnames[3] == expected_columns[3]


def test_point_source_image_with_and_without_variability(quasar_lens_pop_instance):
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    transf_matrix = np.array([[0.2, 0], [0, 0.2]])
    image_data = point_source_coordinate_properties(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        transform_pix2angle=transf_matrix,
    )
    number = len(image_data["image_pix"])
    path = os.path.dirname(__file__)

    psf_image_1 = [np.load(os.path.join(path, "TestData/psf_kernels_for_image_1.npy"))]
    psf_kernels = psf_image_1[:-1]
    psf_kernels.extend([psf_image_1[-1]] * number)

    time = 10
    # Call the function to get the result
    result1 = point_source_image_at_time(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        psf_kernels=psf_kernels,
        transform_pix2angle=transf_matrix,
        time=time,
    )
    result2 = point_source_image_without_variability(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        psf_kernels=psf_kernels,
        transform_pix2angle=transf_matrix,
    )

    transform_matrix = np.array(
        [
            np.array([[0.2, 0], [0, 0.2]]),
            np.array([[0.2, 0], [0, 0.2]]),
            np.array([[0.2, 0], [0, 0.2]]),
            np.array([[0.2, 0], [0, 0.2]]),
        ]
    )
    psf_kernel_1 = psf_image_1[:-1]
    psf_kernel_1.extend([psf_image_1[-1]] * number)
    psf_kernel_2 = psf_image_1[:-1]
    psf_kernel_2.extend([psf_image_1[-1]] * number)
    psf_kernel_3 = psf_image_1[:-1]
    psf_kernel_3.extend([psf_image_1[-1]] * number)

    psf_kernels_all = np.array([psf_kernel_1, psf_kernel_2, psf_kernel_3])
    mag_zero_points = np.array([27, 27, 27])
    t_obs = np.array([10, 20, 30])
    result3 = point_source_image_with_variability(
        lens_class=lens_class,
        band="i",
        mag_zero_point=mag_zero_points,
        delta_pix=0.2,
        num_pix=101,
        psf_kernels=psf_kernels_all,
        transform_pix2angle=transform_matrix,
        t_obs=t_obs,
    )

    assert len(result1) == len(lens_class.point_source_magnification())
    assert len(result2) == len(lens_class.point_source_magnification())
    assert len(result3[0]) == len(t_obs)


if __name__ == "__main__":
    pytest.main()
