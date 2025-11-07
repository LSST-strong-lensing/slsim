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
    convert_mjd_to_days,
    transient_event_time_mjd,
    downsample_galaxies,
    galaxy_size_redshift_evolution,
    flux_error_to_magnitude_error,
    additional_poisson_noise_with_rescaled_coadd,
    additional_bkg_rms_with_rescaled_coadd,
    degrade_coadd_data,
    galaxy_size,
    detect_object,
    surface_brightness_reff,
    gaussian_psf,
    update_cosmology_in_yaml_file,
    draw_coord_in_circle,
)
from slsim.Sources.SourceVariability.variability import Variability
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import FlatLambdaCDM, default_cosmology
import tempfile
import pytest


def test_draw_coord_in_circle():
    np.random.seed(42)
    ra, dec = draw_coord_in_circle(area=2, size=1000)
    npt.assert_almost_equal(np.mean(ra), 0, decimal=2)

    ra, dec = draw_coord_in_circle(area=2, size=1)
    assert isinstance(ra, float)


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
    image = np.load(os.path.join(path, "../TestData/image.npy"))
    psf = np.load(os.path.join(path, "../TestData/psf_kernels_for_deflector.npy"))
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
            [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08]
            * u.rad,
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
            [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08]
            * u.rad,
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


def test_convert_mjd_to_days():
    result = convert_mjd_to_days(60100, 60000)
    assert result == 100


def test_start_point_mjd_time():
    result = transient_event_time_mjd(60000, 60400)
    assert 60000 <= result <= 60400


def test_downsample_galaxies():
    # Create a mock galaxy population
    np.random.seed(42)  # For reproducibility
    galaxy_pop = Table(
        {
            "mag_i": np.random.uniform(18, 28, 1000),  # Magnitudes from 18 to 28
            "z": np.random.uniform(0.1, 1.5, 1000),  # Redshifts from 0.1 to 1.5
        }
    )

    # Define test parameters
    dN = [50, 30, 20]  # Reference galaxy counts per magnitude bin
    dM = 2.0  # Magnitude bin width
    M_min = 18.0  # Minimum magnitude
    M_max = 24.0  # Maximum magnitude
    z_min = 0.3  # Minimum redshift
    z_max = 1.0  # Maximum redshift

    # Run the function
    downsampled_pop = downsample_galaxies(
        galaxy_pop, dN, dM, M_min, M_max, z_min, z_max
    )

    # Assertions
    # Check that the output is an Astropy Table
    assert isinstance(downsampled_pop, Table)

    # Check that the redshifts are within the specified range
    assert np.all((downsampled_pop["z"] > z_min) & (downsampled_pop["z"] <= z_max))
    # Check that the number of galaxies in each magnitude bin matches dN (or is less
    #  if insufficient galaxies exist)
    M_bins = np.arange(M_min, M_max + dM, dM)
    for i in range(len(dN)):
        mask = (downsampled_pop["mag_i"] >= M_bins[i]) & (
            downsampled_pop["mag_i"] < M_bins[i + 1]
        )
        assert len(downsampled_pop[mask]) <= dN[i]

    # Check edge cases
    # Case: No galaxies in the input population within the redshift range
    empty_pop = galaxy_pop[galaxy_pop["z"] < z_min]
    downsampled_empty = downsample_galaxies(
        empty_pop, dN, dM, M_min, M_max, z_min, z_max
    )
    assert len(downsampled_empty) == 0

    # Case: No galaxies in a specific magnitude bin
    dN_with_zero = [50, 0, 20]  # Second bin has zero reference count
    downsampled_zero_bin = downsample_galaxies(
        galaxy_pop, dN_with_zero, dM, M_min, M_max, z_min, z_max
    )
    mask_zero_bin = (downsampled_zero_bin["mag_i"] >= M_bins[1]) & (
        downsampled_zero_bin["mag_i"] < M_bins[2]
    )
    assert len(downsampled_zero_bin[mask_zero_bin]) == 0


def test_galaxy_size_redshift_evolution():
    results = galaxy_size_redshift_evolution(z=0)
    assert results == 4.89


def test_flux_error_to_magnitude_error_basic():
    flux_mean = 100.0
    flux_error = 10.0
    mag_zero_point = 27.0

    mag_mean, mag_error_lower, mag_error_upper = flux_error_to_magnitude_error(
        flux_mean=flux_mean,
        flux_error=flux_error,
        mag_zero_point=mag_zero_point,
        noise=False,
        symmetric=False,
    )

    expected_mag_mean = amplitude_to_magnitude(flux_mean, mag_zero_point)
    assert np.isclose(mag_mean, expected_mag_mean), "Mag mean computation failed."

    upper_flux_limit = flux_mean + flux_error
    lower_flux_limit = max(flux_mean - flux_error, flux_mean * 0.01)

    expected_mag_error_lower = expected_mag_mean - amplitude_to_magnitude(
        upper_flux_limit, mag_zero_point
    )
    expected_mag_error_upper = (
        amplitude_to_magnitude(lower_flux_limit, mag_zero_point) - expected_mag_mean
    )

    assert mag_error_lower == expected_mag_error_lower
    assert mag_error_upper == expected_mag_error_upper


def test_flux_error_to_magnitude_error_symmetric():
    flux_mean = 100.0
    flux_error = 10.0
    mag_zero_point = 27.0

    mag_mean, mag_error_lower, mag_error_upper = flux_error_to_magnitude_error(
        flux_mean=flux_mean,
        flux_error=flux_error,
        mag_zero_point=mag_zero_point,
        noise=False,
        symmetric=True,
    )

    expected_mag_mean = amplitude_to_magnitude(flux_mean, mag_zero_point)
    expected_mag_error = (2.5 / np.log(10)) * flux_error / flux_mean

    assert mag_mean == expected_mag_mean
    assert mag_error_lower == expected_mag_error
    assert mag_error_upper == expected_mag_error


def test_flux_error_to_magnitude_error_with_noise():
    flux_mean = 100.0
    flux_error = 10.0
    mag_zero_point = 27.0

    mag_mean, mag_error_lower, mag_error_upper = flux_error_to_magnitude_error(
        flux_mean=flux_mean,
        flux_error=flux_error,
        mag_zero_point=mag_zero_point,
        noise=True,
        symmetric=False,
    )

    # Noise makes mag_mean non-deterministic, so we validate the magnitude range
    mag_mean_2 = amplitude_to_magnitude(flux_mean, mag_zero_point)

    assert (
        mag_mean_2 - 3 * mag_error_lower < mag_mean <= mag_mean_2 + 3 * mag_error_upper
    )


def test_flux_error_to_magnitude_error_negative_flux():
    mag_mean, mag_error_lower, mag_error_upper = flux_error_to_magnitude_error(
        flux_mean=10.0,
        flux_error=11.0,
        mag_zero_point=27.0,
        symmetric=False,
        noise=False,
    )
    expected_upper_mag = amplitude_to_magnitude(0, 27)
    expected_upper_error = expected_upper_mag - mag_mean
    assert expected_upper_error == mag_error_upper


def test_additional_poisson_noise_with_rescaled_coadd():
    image = np.random.rand(41, 41) * 5
    original_exp_time = np.ones((41, 41))
    degraded_exp_time = original_exp_time * 0.5

    result1 = additional_poisson_noise_with_rescaled_coadd(
        image, original_exp_time, degraded_exp_time, use_noise_diff=True
    )
    result2 = additional_poisson_noise_with_rescaled_coadd(
        image, original_exp_time, degraded_exp_time, use_noise_diff=False
    )

    assert result1.shape == image.shape
    assert np.mean(result1) < np.mean(image)
    assert result2.shape == image.shape
    assert np.mean(result2) < np.mean(image)


def test_additional_bkg_rms_with_rescaled_coadd():
    image = np.random.rand(41, 41) * 5

    result1 = additional_bkg_rms_with_rescaled_coadd(
        image, original_rms=0.5, degraded_rms=0.7, use_noise_diff=True
    )
    result2 = additional_bkg_rms_with_rescaled_coadd(
        image, original_rms=0.5, degraded_rms=0.7, use_noise_diff=False
    )

    assert result1.shape == image.shape
    assert (
        -3 * np.sqrt(0.7**2 - 0.5**2)
        <= np.mean(result1)
        <= 3 * np.sqrt(0.7**2 - 0.5**2)
    )
    assert result2.shape == image.shape
    assert (
        -3 * np.sqrt(0.7**2 - 0.5**2)
        <= np.mean(result2)
        <= 3 * np.sqrt(0.7**2 - 0.5**2)
    )


def test_degrade_coadd_data():
    image = np.random.rand(41, 41) * 5
    variance_map = np.random.rand(41, 41)
    exposure_map = np.ones((41, 41)) * 300
    result = degrade_coadd_data(
        image, variance_map, exposure_map, original_num_years=5, degraded_num_years=1
    )
    assert len(result) == 3
    assert np.mean(image) > np.mean(result[0])


def test_galaxy_size():
    # Define test inputs
    mapp = 24.0  # Apparent g-band magnitude
    zsrc = 0.5  # Source redshift

    # Call function
    Reff, Reff_arcsec = galaxy_size(mapp, zsrc, cosmo)

    # Check outputs are finite and positive
    assert np.isfinite(Reff) and Reff > 0
    assert np.isfinite(Reff_arcsec) and Reff_arcsec > 0
    npt.assert_almost_equal(Reff, 0.5322278567954598, decimal=8)
    npt.assert_almost_equal(Reff_arcsec, 0.08460152399994486, decimal=8)


def test_detect_object():
    path = os.path.dirname(__file__)

    image = np.load(os.path.join(path, "../TestData/psf_kernels_for_image_1.npy")) + 0.5
    variance_map1 = np.abs(np.random.normal(loc=0.1, scale=0.01, size=(57, 57)))
    variance_map2 = np.abs(np.random.normal(loc=0.1, scale=0.01, size=(57, 57)))
    std_dev_map = np.sqrt(variance_map1)
    noise = np.random.normal(loc=0, scale=std_dev_map)
    image1 = image + noise
    result1 = detect_object(image1, variance_map2)
    result2 = detect_object(noise, variance_map2)
    assert result1
    assert not result2


def test_surface_brightness_reff():
    kwargs_source = [
        {
            "magnitude": 15,
            "R_sersic": 1,
            "n_sersic": 1.0,
            "e1": 0.06350855238708408,
            "e2": -0.08420760408362458,
            "center_x": 0.30298310338567075,
            "center_y": -0.3505004565139597,
        }
    ]
    source_model_list = ["SERSIC_ELLIPSE"]
    angular_size = 1
    mag_arcsec2 = surface_brightness_reff(
        angular_size=angular_size,
        source_model_list=source_model_list,
        kwargs_extended_source=kwargs_source,
    )
    npt.assert_almost_equal(mag_arcsec2, 16.995, decimal=2)


def test_gaussian_psf():
    psf_kernel = gaussian_psf(fwhm=0.9, delta_pix=0.2, num_pix=21)
    assert psf_kernel.shape[0] == 21
    npt.assert_almost_equal(np.sum(psf_kernel), 1, decimal=16)


def test_update_cosmology_in_yaml_file():
    # Sample input YAML content with a placeholder cosmology
    original_yaml = """
    simulation:
      name: test_sim
    cosmology: !astropy.cosmology.default_cosmology.get []
    """

    # Create a custom cosmology (different from default)
    custom_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    # Make sure it's not the default cosmology
    assert custom_cosmo != default_cosmology.get()

    # Run the function
    updated_yaml = update_cosmology_in_yaml_file(
        cosmo=custom_cosmo, yml_file=original_yaml
    )

    # Check that some expected parameters are present
    assert "H0:" in updated_yaml
    assert "Om0:" in updated_yaml
    assert "Tcmb0:" in updated_yaml


if __name__ == "__main__":
    pytest.main()
