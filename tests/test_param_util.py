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
    magnitude_to_amplitude,
    amplitude_to_magnitude,
    ellipticity_slsim_to_lenstronomy,
    fits_append_table,
    catalog_with_angular_size_in_arcsec,
)
from slsim.Sources.SourceVariability.variability import Variability
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import tempfile
import pytest


def test_epsilon2e():
    e = epsilon2e(0)
    assert e == 0
    with pytest.raises(ValueError):
        epsilon2e(17)


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
        [0.0, 0.0, 0.0, abs(np.sin(np.pi * np.pi) - 0.0) * 0.5 / (np.pi - 1.0)]
    )
    expect_image_snapshots[:, 0, 1] = np.array(
        [
            0.0,
            0.0,
            0.0,
            abs(2.0 * np.sin(2.0 * np.pi * np.pi) - 0.0) * 0.5 / (np.pi - 1.0),
        ]
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


def test_amplitude_to_magnitude():
    low_flux = 10
    high_flux = 1000
    zero_point = 100
    low_mag = amplitude_to_magnitude(low_flux, zero_point)
    high_mag = amplitude_to_magnitude(high_flux, zero_point)
    assert high_mag < low_mag
    # Test that a constant increasing amplitude makes a constant
    # decreasing magnitude
    fluxes = np.linspace(10**3, 10**5, 50)
    delta_fluxes = fluxes[1:] - fluxes[:-1]
    assert all(delta_fluxes > 0)
    magnitudes = amplitude_to_magnitude(fluxes, zero_point)
    delta_magnitudes = magnitudes[1:] - magnitudes[:-1]
    assert all(delta_magnitudes < 0)


def test_magnitude_to_amplitude():
    low_mag = 23
    high_mag = 21
    zero_point = 20
    low_flux = magnitude_to_amplitude(low_mag, zero_point)
    high_flux = magnitude_to_amplitude(high_mag, zero_point)
    assert high_flux > low_flux
    # Test that this is the inverse of amplitude_to_magnitude()
    new_low_mag = amplitude_to_magnitude(low_flux, zero_point)
    new_high_mag = amplitude_to_magnitude(high_flux, zero_point)
    assert low_mag == new_low_mag
    assert high_mag == new_high_mag


def test_ellipricity_slsim_to_lenstronomy():
    result = ellipticity_slsim_to_lenstronomy(-0.17, 0.05)
    assert result[0] == 0.17
    assert result[1] == 0.05


@pytest.fixture
def temp_fits_file():
    # Create a temporary FITS file
    temp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    temp_file.close()

    # Create an initial empty FITS file
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(temp_file.name, overwrite=True)

    yield temp_file.name

    # Cleanup: Remove the temporary FITS file
    os.remove(temp_file.name)


@pytest.fixture
def sample_table():
    # Create a sample Astropy Table
    return Table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, names=["col1", "col2"])


def test_append_table(temp_fits_file, sample_table):
    # Append the sample table to the FITS file
    fits_append_table(temp_fits_file, sample_table)

    # Read back the FITS file and check the appended table
    with fits.open(temp_fits_file) as hdul:
        # Verify that a new table has been appended
        assert len(hdul) == 2

        # Verify the content of the appended table
        appended_table_data = hdul[1].data
        for i in range(len(sample_table)):
            assert appended_table_data["col1"][i] == sample_table["col1"][i]
            assert appended_table_data["col2"][i] == sample_table["col2"][i]


def test_catalog_with_angular_size_in_arcsec():
    galaxy_list = Table(
        [
            [0.5, 0.5, 0.5],
            [-15.248975044343094, -15.248975044343094, -15.248975044343094],
            [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
            [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08]*u.rad,
            [23, 23, 23],
            [43, 43, 43],
        ],
        names=("z", "M", "e", "angular_size", "mag_i", "a_rot"),
    )
    galaxy_list2 = Table(
        [
            [0.5, 0.5, 0.5],
            [-15.248975044343094, -15.248975044343094, -15.248975044343094],
            [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
            [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08]*u.rad,
            [23, 23, 23],
            [43, 43, 43],
        ],
        names=("z", "M", "e", "angular_size", "mag_i", "a_rot"),
    )
    galaxy_cat = catalog_with_angular_size_in_arcsec(
        galaxy_catalog=galaxy_list, input_catalog_type="skypy"
    )
    galaxy_cat2 = catalog_with_angular_size_in_arcsec(
        galaxy_catalog=galaxy_list2, input_catalog_type="other"
    )
    assert galaxy_cat["angular_size"][0] == 4.186996407348755e-08 / 4.84813681109536e-06
    assert galaxy_cat2["angular_size"][0] == 4.186996407348755e-08
    assert galaxy_cat2["angular_size"].unit == u.rad
    assert galaxy_cat["angular_size"].unit == u.arcsec


if __name__ == "__main__":
    pytest.main()
