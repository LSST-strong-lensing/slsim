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
    deflector_images_with_different_zeropoint,
    image_plus_poisson_noise,
    lens_image,
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


@pytest.fixture
def galaxy_lens_pop_instance():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = Quantity(value=0.1, unit="deg2")

    return LensPop(
        deflector_type="all-galaxies",
        source_type="galaxies",
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
    keys = result.keys()
    result_key = []
    for key in keys:
        result_key.append(key)

    expected_key = ["deflector_pix", "image_pix", "ra_image", "dec_image"]
    assert result_key[0] == expected_key[0]
    assert result_key[1] == expected_key[1]
    assert result_key[2] == expected_key[2]
    assert result_key[3] == expected_key[3]


def test_point_source_image_with_and_without_variability(quasar_lens_pop_instance):
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    transf_matrix = np.array([[0.2, 0], [0, 0.2]])
    path = os.path.dirname(__file__)

    psf_image_1 = [np.load(os.path.join(path, "TestData/psf_kernels_for_image_1.npy"))]
    psf_kernel_single = psf_image_1[-1]

    # Call the function to get the result
    result1 = point_source_image_at_time(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        psf_kernel=psf_kernel_single,
        transform_pix2angle=transf_matrix,
        time=10,
    )
    result2 = point_source_image_without_variability(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=101,
        psf_kernel=psf_kernel_single,
        transform_pix2angle=transf_matrix,
    )

    transform_matrix = np.array(
        [
            np.array([[0.2, 0], [0, 0.2]]),
            np.array([[0.2, 0], [0, 0.2]]),
            np.array([[0.2, 0], [0, 0.2]]),
        ]
    )
    mag_zero_points = np.array([27, 27, 27])
    t_obs = np.array([10, 20, 30])
    psf_kernels = psf_image_1[:-1]
    psf_kernels.extend([psf_image_1[-1]] * 3)
    result3 = point_source_image_with_variability(
        lens_class=lens_class,
        band="i",
        mag_zero_point=mag_zero_points,
        delta_pix=0.2,
        num_pix=101,
        psf_kernels=psf_kernels,
        transform_pix2angle=transform_matrix,
        t_obs=t_obs,
    )

    assert len(result1) == len(lens_class.point_source_magnification())
    assert len(result2) == len(lens_class.point_source_magnification())
    assert len(result3) == len(t_obs)


def test_deflector_images_with_different_zeropoint(quasar_lens_pop_instance):
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    lens_class = quasar_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    mag_zero_points = np.array([27, 30])
    result_images = deflector_images_with_different_zeropoint(
        lens_class=lens_class,
        band="i",
        mag_zero_point=mag_zero_points,
        delta_pix=0.2,
        num_pix=64,
    )
    noise_image = image_plus_poisson_noise(result_images[0], exposure_time=30)
    diff_image = noise_image - result_images[0]
    exposure_time = [20, 30]
    result_list = image_plus_poisson_noise(result_images, exposure_time=exposure_time)
    path = os.path.dirname(__file__)

    psf_image_1 = [np.load(os.path.join(path, "TestData/psf_kernels_for_image_1.npy"))]
    psf_kernel_single = psf_image_1[-1]
    transf_matrix = np.array([[0.2, 0], [0, 0.2]])
    lens_image_result_1 = lens_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=64,
        psf_kernels=psf_kernel_single,
        transform_pix2angle=transf_matrix,
        exposure_time=30,
        t_obs=None,
    )
    lens_image_result_2 = lens_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=27,
        delta_pix=0.2,
        num_pix=64,
        psf_kernels=psf_kernel_single,
        transform_pix2angle=transf_matrix,
        exposure_time=None,
        t_obs=None,
    )
    diff_lens_image_result_3 = lens_image_result_2 - lens_image_result_1

    transform_matrix = np.array(
        [
            np.array([[0.2, 0], [0, 0.2]]),
            np.array([[0.2, 0], [0, 0.2]]),
        ]
    )
    psf_kernels = psf_image_1[:-1]
    psf_kernels.extend([psf_image_1[-1]] * 2)
    lens_image_result_4 = lens_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=np.array([31, 30]),
        delta_pix=0.2,
        num_pix=64,
        psf_kernels=psf_kernels,
        transform_pix2angle=transform_matrix,
        exposure_time=np.array([30, 27]),
        t_obs=np.array([20, 40]),
    )
    lens_image_result_5 = lens_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=np.array([31, 30]),
        delta_pix=0.2,
        num_pix=64,
        psf_kernels=psf_kernels,
        transform_pix2angle=transform_matrix,
        exposure_time=None,
        t_obs=np.array([20, 40]),
    )
    diff_lens_image_result_6 = lens_image_result_4[0] - lens_image_result_5[0]

    assert len(result_images) == len(mag_zero_points)
    assert np.any(diff_image != 0) == True
    assert len(result_list) == len(result_images)
    assert lens_image_result_1.shape[0] == 64
    assert np.any(diff_lens_image_result_3 != 0) == True
    assert len(lens_image_result_4) == 2
    assert np.any(diff_lens_image_result_6 != 0) == True


def test_lens_image_extended(galaxy_lens_pop_instance):
    kwargs_lens_cut = {"min_image_separation": 0.8, "max_image_separation": 10}
    lens_class = galaxy_lens_pop_instance.select_lens_at_random(**kwargs_lens_cut)
    path = os.path.dirname(__file__)
    psf_image_1 = [np.load(os.path.join(path, "TestData/psf_kernels_for_image_1.npy"))]
    psf_kernel_single = psf_image_1[-1]
    transf_matrix = np.array([[0.2, 0], [0, 0.2]])
    lens_image_result_1 = lens_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=30,
        delta_pix=0.2,
        num_pix=64,
        psf_kernels=psf_kernel_single,
        transform_pix2angle=transf_matrix,
        exposure_time=30,
        t_obs=None,
    )
    lens_image_result_2 = lens_image(
        lens_class=lens_class,
        band="i",
        mag_zero_point=30,
        delta_pix=0.2,
        num_pix=64,
        psf_kernels=psf_kernel_single,
        transform_pix2angle=transf_matrix,
        exposure_time=None,
        t_obs=None,
    )

    diff = lens_image_result_1 - lens_image_result_2
    assert lens_image_result_1.shape[0] == 64
    assert np.any(diff != 0) == True
    with pytest.raises(ValueError) as excinfo:
        lens_image(
            lens_class=lens_class,
            band="i",
            mag_zero_point=30,
            delta_pix=0.2,
            num_pix=64,
            psf_kernels=psf_kernel_single,
            transform_pix2angle=transf_matrix,
            exposure_time=None,
            t_obs=30,
        )
    assert (
        "extented source do not have time variability. So,"
        "do not provide observation time."
    ) in str(excinfo.value)


if __name__ == "__main__":
    pytest.main()
