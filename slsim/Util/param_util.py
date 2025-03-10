import numpy as np
import scipy
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from lenstronomy.Util.param_util import transform_e1e2_product_average
from lenstronomy.Util.param_util import ellipticity2phi_q
from astropy.io import fits
from astropy import units as u
from astropy.stats import sigma_clipped_stats
import warnings


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


def ellip_from_axis_ratio2epsilon(ellip):
    """Translates ellipticity definitions from.

    .. math::
        ellip = \\equic \\1 - q

    to

    .. math::
        epsilon = \\equic \\frac{1 - q^2}{1 + q^2}

    :param ellip: ellipticity in SL-Hammocks
    :type  ellip: ndarray or float
    :return: epsilon. ellipticity in slsim
    """
    return (1.0 - (1.0 - ellip) ** 2) / (1.0 + (1.0 - ellip) ** 2)


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
    :param psf_kernel: kernel used to convolve the given image. It
        should be a pixel psf kernel.
    :param convolution_type: method to be used to convolve image.
        currently fftconvolve and convolve2d are supported. The default
        type is fftconvolve and we prefer to use fftconvolve over
        convolve2d because it is relatively faster for our purpose.
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
    """Converts a string of pixel snapshots back into a series of image
    snapshots. This is the inverse of images_to_pixels.

    :param pixels: Series of pixel snapshots to arrange back into the
        original image shape
    :param original_shape: The original output of
        np.shape(original_image_series) [tuple]
    :return: Series of image snapshots
    """
    return np.reshape(
        pixels, (np.size(pixels, 0), original_shape[1], original_shape[2])
    )


def interpolate_variability(image_series, orig_timestamps, new_timestamps):
    """Interpolates between time stamps of a series of image snapshots. This
    will be important for future implimentation of microlensing.

    :param image_series: 3 dimensional array of shape (snapshot_index,
        x, y) defining snapshots of a variable object in (x, y)
        coordinates
    :param orig_timestamps: List of timestamps which represent the time
        of each simulated observation, must be same length as
        np.size(image_series, 0)
    :param new_timestamps: List of new timestamps to interpolate the
        series of snapshots to
    :return: Linearly interpolated series of snapshots at the new
        timestamps on a pixel- by-pixel basis
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

    :param tranform_matrix: transformation matrix (2x2) of pixels into
        coordinate displacements
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


def deg2_to_cone_angle(solid_angle_deg2):
    """Convert solid angle from square degrees to half cone angle in radians.

    This function translates a solid angle, specified in square degrees,
    into the corresponding half cone angle expressed in radians. This
    conversion is essential for applications involving angular
    measurements in astronomy, particularly in lensing calculations
    where the geometry of observations is defined in terms of cone
    angles.

    :param solid_angle_deg2: Solid angle in square degrees to be
        converted.
    :type solid_angle_deg2: float
    :return: The half cone angle in radians equivalent to the input
        solid angle.
    :rtype: float :note: The conversion utilizes the relationship
        between solid angles in steradians and the apex angle of a cone,
        facilitating a direct transition from square degrees to radians.
    """

    solid_angle_sr = solid_angle_deg2 * (np.pi / 180) ** 2
    theta = np.arccos(1 - solid_angle_sr / (2 * np.pi))  # rad
    return theta


def ellipticity_slsim_to_lenstronomy(e1_slsim, e2_slsim):
    """Converts ellipticity component from slsim convension to lenstronomy
    convention. In slsim, position angle goes from North to East. In
    lenstronomy, position angle goes from East to North.

    :param e1_slsim:
        first component of the ellipticity in slsim convension i.e position
        angle from north to east.

    :param e2_slsim:
        second component of the ellipticity in slsim convention.

    return: ellipticity components in lenstronomy convention.
    """

    return -e1_slsim, e2_slsim


def elliptical_distortion_product_average(x, y, e1, e2, center_x, center_y):
    """Maps the coordinates x, y with eccentricities e1, e2 into a new
    elliptical coordinate system with same coordinate orientation.

    :param x: x-coordinate
    :param y: y-coordinate
    :param e1: eccentricity
    :param e2: eccentricity
    :param center_x: center of distortion
    :param center_y: center of distortion
    :return: distorted coordinates x', y'
    """
    x_, y_ = transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)

    # Rotate back
    phi_g, q = ellipticity2phi_q(e1, e2)
    cos_phi = np.cos(-phi_g)
    sin_phi = np.sin(-phi_g)

    x__ = cos_phi * x_ + sin_phi * y_
    y__ = -sin_phi * x_ + cos_phi * y_

    # Shift
    x___ = x__ + center_x
    y___ = y__ + center_y

    return x___, y___


def fits_append_table(filename, table):
    """Append an Astropy Table to an existing FITS file.

    :param filename: Name of the FITS file to append to
    :param table: Astropy Table object to append
    """
    hdulist = fits.open(filename, mode="append")
    hdulist.append(fits.BinTableHDU(table))
    hdulist.writeto(filename, overwrite=True)
    hdulist.close()


def catalog_with_angular_size_in_arcsec(galaxy_catalog, input_catalog_type="skypy"):
    """This function is written to change unit of angular size in skypy galaxy
    catalog to arcsec. If user is using deflector catalog other than generated
    from skypy pipeline, we require them to provide angular size of the galaxy
    in arcsec.

    :param galaxy_catalog: galaxy catalog.
    :param input_catalog_type: type of the catalog.
    :type input_catalog_type: str. "skypy" or None
    :return: galaxy catalog with anularsize in arcsec.
    """
    copied_galaxy_catalog = galaxy_catalog.copy()
    if input_catalog_type == "skypy":
        copied_galaxy_catalog["angular_size"] = copied_galaxy_catalog[
            "angular_size"
        ].to(u.arcsec)
        warning_msg = (
            "Angular size is converted to arcsec because provided"
            " input_catalog_type is skypy. If this is not correct, please refer to"
            " the documentation of the class you are using"
        )
        warnings.warn(warning_msg, category=UserWarning, stacklevel=2)
    else:
        warning_msg = (
            "You provided angular size in arcsec. If this is not correct, please"
            " refer to the documentation of the class that you are using"
        )
        warnings.warn(warning_msg, category=UserWarning, stacklevel=2)
    return copied_galaxy_catalog


def convert_mjd_to_days(reference_mjd, start_point_mjd):
    """Convert reference MJD(s) to days relative to a chosen zero-point MJD.

    :param reference_mjd: The reference MJD(s) to convert.
    :type reference_mjd: float, list, or numpy.ndarray
    :param start_point_mjd: The zero-point MJD to use as the reference.
    :return: The time(s) in days relative to the zero-point MJD.
    """
    # Ensure input is a NumPy array for consistent handling
    reference_mjd = np.array(reference_mjd)
    return reference_mjd - start_point_mjd


def transient_event_time_mjd(min_mjd, max_mjd):
    """Produces a random MJD time with in the given range.

    :param min_mjd: Minimum bound for the MJD time
    :param max_mjd: Maximum bound for the MJD time
    :return: A random MJD time between given min and max bounds.
    """
    start_mjd = np.random.randint(min_mjd, max_mjd)
    return start_mjd


def downsample_galaxies(galaxy_pop, dN, dM, M_min, M_max, z_min, z_max):
    """Downsamples a galaxy population to match the luminosity function of
    another population. Another population with the given redshift range is
    specified by the dN.

    :param galaxy_population: astropy.table.Table. Table containing the
        galaxy population with at least a 'magnitude' column.
    :param dN: array-like. Galaxy counts per magnitude bin for the
        reference population.
    :param dM: float. Magnitude bin width.
    :param M_min: float. Minimum magnitude for binning.
    :param M_max: float. Maximum magnitude for binning.
    :param z_min: float. Minimum redshift for sample.
    :param z_max: float. Maximum redshift for sample.
    :returns: astropy.table.Table. Downsampled galaxy population.
    """
    galaxy_pop = galaxy_pop[(galaxy_pop["z"] > z_min) & (galaxy_pop["z"] <= z_max)]
    # Create magnitude bins
    M_bins = np.arange(M_min, M_max + dM, dM)

    # Downsample second population to match dN
    selected_indices = []

    for i in range(len(dN)):
        # Identify galaxies in the current magnitude bin
        mask = (galaxy_pop["mag_i"] >= M_bins[i]) & (
            galaxy_pop["mag_i"] < M_bins[i + 1]
        )
        indices = np.where(mask)[0]

        # Determine how many galaxies to keep
        N_to_keep = min(dN[i], len(indices))

        # Randomly select indices
        if N_to_keep > 0 and len(indices) > 0:
            selected = np.random.choice(indices, N_to_keep, replace=False)
            selected_indices.extend(selected)

    # Create a downsampled population
    downsampled_pop = galaxy_pop[selected_indices]
    return downsampled_pop


def vel_disp_from_m_star(m_star):
    """Function to calculate the velocity dispersion from the staller mass
    using empirical relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::

         V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
         M_\\odot} \\right)^{0.24}

    2.32,0.24 is the parameters from [1] table 2
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    :param m_star: stellar mass in the unit of solar mass
    :return: the velocity dispersion ("km/s")
    """
    v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)
    return v_disp


def galaxy_size_redshift_evolution(z):
    """This function provides a galaxy size elolution with the redshift.
    Provides a correction factor to the size relative to z=0.
    This relation is taken from Shibuya et al. (2015): https://arxiv.org/abs/1503.07481

    :param z: galaxy redshift. It can be a single galaxy redshift or list of galaxy
     redshifts.
    :return: Physical size of the galaxy.
    """
    Bz = 4.89  # median value from Table 6 of Shibuya et al. (2015)
    betaz = -1.05  # median value from Table 6 of Shibuya et al. (2015)
    return Bz * (1 + z) ** betaz


def additional_poisson_noise_with_rescaled_coadd(
    image, original_exp_time, degraded_exp_time, use_noise_diff=True
):
    """Computes additional Poisson noise to an image based on the change in
    exposure time.

    :param image : numpy.ndarray The input image array.
    :param original_exp_time : numpy.ndarray The original exposure time
        per pixel.
    :param degraded_exp_time : numpy.ndarray The degraded exposure time
        per pixel.
    :param use_noise_diff : bool, optional If True, approximates noise
        difference using Gaussian noise, otherwise, applies Poisson
        sampling. Default is True.
    :return: numpy.ndarray The additional noise to be added to the
        image.
    """
    image_positive = np.where(image > 0, image, 0)

    if use_noise_diff:
        sigma_add = np.where(
            original_exp_time > 0,
            np.sqrt(image_positive)
            * np.sqrt(1 / degraded_exp_time - 1 / original_exp_time),
            0.0,
        )
        return np.random.normal(scale=sigma_add, size=image.shape)
    else:
        image_with_poisson = np.where(
            original_exp_time > 0,
            np.random.poisson(lam=image_positive * degraded_exp_time)
            / degraded_exp_time,
            0.0,
        )
        return image_with_poisson - image_positive


def additional_bkg_rms_with_rescaled_coadd(
    image, original_rms, degraded_rms, use_noise_diff=True
):
    """Computes additinal background noise based on RMS values before and after
    degradation.

    :param image : numpy.ndarray The input image array.
    :param original_rms : float The original root mean square (RMS)
        noise.
    :param degraded_rms : float The degraded RMS noise.
    :param use_noise_diff : bool, optional If True, approximates noise
        difference using Gaussian noise, otherwise, applies new Gaussian
        noise directly. Default is True.
    :return: numpy.ndarray The additional noise to be added to the
        image.
    """
    if use_noise_diff:
        sigma_add = np.sqrt(degraded_rms**2 - original_rms**2)
        return np.random.normal(scale=sigma_add, size=image.shape)
    else:
        return np.random.normal(scale=degraded_rms, size=image.shape)


def degrade_coadd_data(
    image,
    variance_map,
    exposure_map,
    original_num_years=5,
    degraded_num_years=1,
    use_noise_diff=True,
):
    """Degrade a coadded astronomical image by reducing its effective exposure
    time.

    :param image : numpy.ndarray The input image array.
    :param variance_map : numpy.ndarray The original variance map.
    :param exposure_map : numpy.ndarray The original exposure time per
        pixel.
    :param original_num_years : int, optional The original coadded
        number of years. Default is 5.
    :param degraded_num_years : int, optional The new degraded number of
        years. Default is 1.
    :param use_noise_diff : bool, optional If True, approximates noise
        difference using Gaussian noise, otherwise, applies full noise
        resampling. Default is True.
    :return: The degraded image, the new variance map, and he new
        exposure map.
    """
    degraded_var_map = variance_map * original_num_years / degraded_num_years
    degraded_exp_map = exposure_map * degraded_num_years / original_num_years

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_rms = np.sqrt(sigma_clipped_stats(variance_map, sigma=3)[0])
        degraded_rms = np.sqrt(sigma_clipped_stats(degraded_var_map, sigma=3)[0])

        degraded_image = image + additional_poisson_noise_with_rescaled_coadd(
            image, exposure_map, degraded_exp_map, use_noise_diff
        )

    degraded_image += additional_bkg_rms_with_rescaled_coadd(
        image, original_rms, degraded_rms, use_noise_diff
    )

    return degraded_image, degraded_var_map, degraded_exp_map

def galaxy_size(mapp, zsrc, cosmo):
    """
    Calculate the half-light radius of a source using the size-luminosity relation
    from Bernardi et al. (2003), as given in Oguri (2006). Please see equation 15 of 
    : https://arxiv.org/pdf/astro-ph/0508528
    
    :param mapp: float
        Apparent g-band magnitude of the source.
    :param zsrc: float
        Redshift of the source.
    :param cosmo: astropy.cosmology instance
    :return: Half-light radius in kpc and arcsec.
    """

    # Compute luminosity distance (in Mpc)
    Dlum = cosmo.luminosity_distance(zsrc).value

    # Compute absolute magnitude
    Mabs = mapp - 5 * np.log10(Dlum) - 25

    # Compute luminosity in solar units (using g-band solar magnitude 5.48)
    Lum_src = 10**(-0.4 * (Mabs - 5.48))

    # Compute angular diameter distance (in kpc)
    Da = cosmo.angular_diameter_distance(zsrc).to(u.kpc).value

    # Compute the effective radius using the size-luminosity relation
    Lrat = Lum_src / 10**10.2
    Reff = (10**0.52) * (Lrat**(2/3))*((0.7/cosmo.h)**(2/3)) / (1 + zsrc)**2  # in kpc

    # Convert kpc to arcsec, then to pixels
    Reff_arcsec = (Reff / Da) * (u.rad.to(u.arcsec))

    return Reff, Reff_arcsec

