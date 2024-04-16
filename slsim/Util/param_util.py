import numpy as np
from scipy.signal import convolve2d, fftconvolve
import scipy


def epsilon2e(epsilon):
    """Translates ellipticity definitions from.

    .. math::
        epsilon = \\equic \\frac{1 - q^2}{1 + q^2}

    to

    .. math::
        e = \\equic \\frac{1 - q}{1 + q}

    :param epsilon: ellipticity
    :return: eccentricity
    """
    if epsilon == 0:
        return 0
    elif 0 < epsilon <= 1:
        return (1 - np.sqrt(1 - epsilon**2)) / epsilon
    else:
        raise ValueError('Value of "epsilon" is %s and needs to be in [0, 1]' % epsilon)


def e2epsilon(e):
    """Translates ellipticity definitions from.

    .. math::
        e = \\equic \\frac{1 - q}{1 + q}

    to

    .. math::
        epsilon = \\equic \\frac{1 - q^2}{1 + q^2}

    :param e: eccentricity
    :return: ellipticity
    """
    return 2 * e / (1 + e**2)


def random_ra_dec(ra_min, ra_max, dec_min, dec_max, n):
    """Generates n number of random ra, dec pair with in a given limits.

    :param ra_min: minimum limit for ra
    :param ra_max: maximum limit for ra
    :param dec_min: minimum limit for dec
    :param dec_max: maximum limit for dec
    :param n: number of random sample
    :returns: n number of ra, dec pair within given limits
    """
    ra = np.random.uniform(ra_min, ra_max, n)
    dec = np.random.uniform(dec_min, dec_max, n)
    return ra, dec


def random_radec_string(ra_min, ra_max, dec_min, dec_max, n):
    """Generates n number of random "ra, dec" string within given limits.

    :param ra_min: minimum limit for ra
    :param ra_max: maximum limit for ra
    :param dec_min: minimum limit for dec
    :param dec_max: maximum limit for dec
    :param n: number of random sample
    :returns: n number of "ra, dec" strings within given limits
    """
    ra, dec = random_ra_dec(
        ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max, n=n
    )
    center_coods_list = []
    for i in range(n):
        center_coods_list.append(str(ra[i]) + ", " + str(dec[i]))
    return center_coods_list


def convolved_image(image, psf_kernel, convolution_type="fft"):
    """Convolves an image with given psf kernel.

    :param image: image to be convolved
    :param psf_kernel: kernel used to convolve the given image. It should be a pixel psf
        kernel.
    :param convolution_type: method to be used to convolve image. currently fftconvolve
        and convolve2d are supported. The default type is fftconvolve and we prefer to
        use fftconvolve over convolve2d because it is relatively faster for our purpose.
    :returns: convolved image.
    """
    if convolution_type == "fft":
        return fftconvolve(image, psf_kernel, mode="same")
    if convolution_type == "grid":
        return convolve2d(
            image, psf_kernel, mode="same", boundary="symm", fillvalue=0.0
        )


def magnitude_to_amplitude(magnitude, mag_zero_point):
    """Converts source magnitude to amplitude.

    :param magnitude: source magnitude
    :param mag_zero_point: zero point magnitude for the image
    :returns: source amplitude in counts per second
    """
    delta_m = magnitude - mag_zero_point
    counts = 10 ** (-delta_m / 2.5)
    return counts


def amplitude_to_magnitude(amplitude, mag_zero_point):
    """Converts source amplitude to magnitude.

    The inverse of     magnitude_to_amplitude().
    :param amplitude: source amplitude in flux
    :param mag_zero_point: zero point magnitude
    :returns: source magnitude
    """
    delta_m = -np.log10(amplitude) * 2.5
    magnitude = delta_m + mag_zero_point
    return magnitude


def images_to_pixels(image_series):
    """Converts a series of image snapshots into a list of pixel snapshots.

    :param image_series: Series of images to convert
    :return: List of pixel snapshots
    """
    initial_shape = np.shape(image_series)
    number_of_pixels = initial_shape[1] * initial_shape[2]
    return np.reshape(image_series, (initial_shape[0], number_of_pixels))


def pixels_to_images(pixels, original_shape):
    """Converts a string of pixel snapshots back into a series of image snapshots. This
    is the inverse of images_to_pixels.

    :param pixels: Series of pixel snapshots to arrange back into the original image
        shape
    :param original_shape: The original output of np.shape(original_image_series)
        [tuple]
    :return: Series of image snapshots
    """
    return np.reshape(
        pixels, (np.size(pixels, 0), original_shape[1], original_shape[2])
    )


def interpolate_variability(image_series, orig_timestamps, new_timestamps):
    """Interpolates between time stamps of a series of image snapshots. This will be
    important for future implimentation of microlensing.

    :param image_series: 3 dimensional array of shape (snapshot_index, x, y) defining
        snapshots of a variable object in (x, y) coordinates
    :param orig_timestamps: List of timestamps which represent the time of each
        simulated observation, must be same length as np.size(image_series, 0)
    :param new_timestamps: List of new timestamps to interpolate the series of snapshots
        to
    :return: Linearly interpolated series of snapshots at the new timestamps on a pixel-
        by-pixel basis
    """
    time_varying_pixels = images_to_pixels(image_series)
    number_pixels = np.size(time_varying_pixels, 1)
    pixel_positions = np.linspace(1, number_pixels, number_pixels)

    # prepare the interpolation grid. bounds_error=False allows for endpoint
    # "interpolation" (e.g. allows t=0 in new_timestamps)
    interpolation = scipy.interpolate.RegularGridInterpolator(
        (orig_timestamps, pixel_positions),
        time_varying_pixels,
        bounds_error=False,
        fill_value=None,
    )

    # define new times to interpolate to, and do not interpolate between pixels
    new_time_points = np.meshgrid(new_timestamps, pixel_positions, indexing="ij")
    pixels_resampled = interpolation((new_time_points[0], new_time_points[1]))
    return pixels_to_images(pixels_resampled, np.shape(image_series))


def transformmatrix_to_pixelscale(tranform_matrix):
    """Calculates pixel scale using tranform matrix.

    :param tranform_matrix: transformation matrix (2x2) of pixels into coordinate
        displacements
    :return: pixel scale
    """
    determinant = np.linalg.det(tranform_matrix)
    return np.sqrt(determinant)


def average_angular_size(a, b):
    """Computes average angular size using semi major and minor axis.

    :param a: value of semi major axis in arcsec
    :param b: value of semi minor axis in arcsec
    :return: average angular size in arcsec
    """
    return np.sqrt(a * b)


def axis_ratio(a, b):
    """Computes axis ratio using semi major and minor axis.

    :param a: value of semi major
    :param b: value of semi minor
    :return: axis ratio
    """
    return b / a


def eccentricity(q):
    """Computes eccentricity using axis ratio.

    :param q: axis ratio of an object
    :return: eccentricity
    """
    return (1 - q) / (1 + q)
