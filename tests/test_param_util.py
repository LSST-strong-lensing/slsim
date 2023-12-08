import numpy as np
import os
from numpy import testing as npt
from slsim.Util.param_util import (
    epsilon2e,
    e2epsilon,
    random_ra_dec,
    convolved_image,
    interpolate_variability,
    images_to_pixels,
    pixels_to_images,
    random_radec_string,
    transformmatrix_to_pixelscale,
)
from slsim.Sources.SourceVariability.variability import Variability
import pytest


def test_epsilon2e():
    e = epsilon2e(0)
    assert e == 0


def test_e2epsilon():
    ep = e2epsilon(0)
    assert ep == 0


def test_random_ra_dec():
    ra, dec = random_ra_dec(ra_min=30, ra_max=62, dec_min=-63, dec_max=-36, n=50)
    assert len(ra) == 50


def test_convolved_image():
    path = os.path.dirname(__file__)
    image = np.load(os.path.join(path, "TestData/image.npy"))
    psf = np.load(os.path.join(path, "TestData/psf_kernels_for_deflector.npy"))
    c_image = convolved_image(image, psf)
    c_image_1 = convolved_image(image, psf, convolution_type="grid")
    assert c_image.shape[0] == 101
    assert c_image_1.shape[0] == 101


def test_images_to_pixels():
    image = np.reshape(np.linspace(1, 27, 27), (3, 3, 3))
    ordered_pixels = images_to_pixels(image)
    manually_ordered_pixels = np.concatenate(
        (image[0, 0, :], image[0, 1, :], image[0, 2, :])
    )
    # Test pixels are properly ordered
    assert all(ordered_pixels[0] == manually_ordered_pixels)
    # Test axis 0 is still the snapshot index
    assert all(ordered_pixels[1] == ordered_pixels[0] + 9)
    assert all(ordered_pixels[2] == ordered_pixels[1] + 9)


def test_pixels_to_images():
    image = np.reshape(np.linspace(1, 27, 27), (3, 3, 3))
    ordered_pixels = images_to_pixels(image)
    image_reconstructed = pixels_to_images(ordered_pixels, np.shape(image))
    # Test image_reconstructed is exactly the original image
    assert np.all(image_reconstructed == image)


def test_interpolation_for_sinusoidal():
    # define original and new timestamps for interpolation.
    # tests: interpolating from boundary point to same boundary point
    #        midpoint between two timestamps
    #        interpolating from a central timestamp to same timestamp
    #        interpolating to arbitrary timestamp (not at midpoint)
    #        output shape is correct
    obs_snapshots = np.array([0, 1, np.pi])
    new_snapshots = np.array([0, 0.5, 1, 1.5])
    # define 3 snapshots of 4 independent, time-varying pixels
    kwargs_model00 = {"amp": 1.0, "freq": 0.5}
    kwargs_model01 = {"amp": 2.0, "freq": 1.0}
    # manually set each pixel's values
    image_snapshots = np.zeros((3, 2, 2))
    image_snapshots[:, 0, 0] = Variability(
        "sinusoidal", **kwargs_model00
    ).variability_at_time(obs_snapshots)
    image_snapshots[:, 0, 1] = Variability(
        "sinusoidal", **kwargs_model01
    ).variability_at_time(obs_snapshots)
    # also define two pixels as simple values, including negative
    image_snapshots[:, 1, 0] = np.array([4, 2, 2])
    image_snapshots[:, 1, 1] = np.array([0, -6, 0])
    # interpolate to new timestamps
    interp_image_snapshots = interpolate_variability(
        image_snapshots, obs_snapshots, new_snapshots
    )
    # manually calculate expectation snapshots
    expect_image_snapshots = np.zeros((4, 2, 2))
    expect_image_snapshots[:, 0, 0] = np.array(
        [0.0, 0.0, 0.0, (np.sin(np.pi * np.pi) - 0.0) * 0.5 / (np.pi - 1.0)]
    )
    expect_image_snapshots[:, 0, 1] = np.array(
        [0.0, 0.0, 0.0, (2.0 * np.sin(2.0 * np.pi * np.pi) - 0.0) * 0.5 / (np.pi - 1.0)]
    )
    expect_image_snapshots[:, 1, 0] = np.array([4.0, 3.0, 2.0, 2.0])
    expect_image_snapshots[:, 1, 1] = np.array(
        [0.0, -3.0, -6.0, -6.0 + (0.0 + 6.0) * 0.5 / (np.pi - 1.0)]
    )
    # compare to 5 decimal points
    npt.assert_almost_equal(interp_image_snapshots, expect_image_snapshots, decimal=5)


def test_random_radec_string():
    radec_result = random_radec_string(
        ra_min=30, ra_max=62, dec_min=-63, dec_max=-36, n=50
    )
    assert len(radec_result) == 50
    assert all(isinstance(item, str) for item in radec_result) is True


def test_transformmatrix_to_pixelscale():
    transform_matrix = np.array([[2, 0], [0, 3]])

    result = transformmatrix_to_pixelscale(transform_matrix)
    expected_result = np.sqrt(6)

    assert result == expected_result


if __name__ == "__main__":
    pytest.main()
