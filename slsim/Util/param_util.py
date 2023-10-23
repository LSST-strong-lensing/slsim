import numpy as np
from scipy.signal import convolve2d, fftconvolve


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


def convolved_image(image, psf_kernel, type="fft"):
    """Convolves an image with given psf kernel.

    :param image: image to be convolved
    :param psf_kernel: kernel used to convolve the given image. It should be a pixel psf
        kernel.
    :param type: method to be used to convolve image. currently fftconvolve and
        convolve2d are supported. The default type is fftconvolve and we prefer to use
        fftconvolve over convolve2d because it is relatively faster for our purpose.
    :returns: convolved image.
    """
    if type == "fft":
        return fftconvolve(image, psf_kernel, mode="same")
    if type == "grid":
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
