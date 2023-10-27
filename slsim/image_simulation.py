import numpy as np
from astropy.table import Table
from lenstronomy.SimulationAPI.sim_api import SimAPI
from astropy.visualization import make_lupton_rgb
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering
from slsim.Util.param_util import magnitude_to_amplitude


def simulate_image(
    lens_class, band, num_pix, add_noise=True, observatory="LSST", **kwargs
):
    """Creates an image of a selected lens with noise.

    :param lens_class: class object containing all information of the lensing system
        (e.g., Lens())
    :param band: imaging band
    :param num_pix: number of pixels per axis
    :param add_noise: if True, add noise
    :param observatory: telescope type to be simulated
    :type observatory: str
    :param kwargs: additional keyword arguments for the bands
    :type kwargs: dict
    :return: simulated image
    :rtype: 2d numpy array
    """
    kwargs_model, kwargs_params = lens_class.lenstronomy_kwargs(band)
    from slsim.Observations import image_quality_lenstronomy

    kwargs_single_band = image_quality_lenstronomy.kwargs_single_band(
        observatory=observatory, band=band, **kwargs
    )

    sim_api = SimAPI(
        numpix=num_pix, kwargs_single_band=kwargs_single_band, kwargs_model=kwargs_model
    )
    kwargs_lens_light, kwargs_source, kwargs_ps = sim_api.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_params.get("kwargs_lens_light", None),
        kwargs_source_mag=kwargs_params.get("kwargs_source", None),
        kwargs_ps_mag=kwargs_params.get("kwargs_ps", None),
    )
    kwargs_numerics = {
        "point_source_supersampling_factor": 1,
        "supersampling_factor": 3,
    }
    image_model = sim_api.image_model_class(kwargs_numerics)
    kwargs_lens = kwargs_params.get("kwargs_lens", None)
    image = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_ps=kwargs_ps,
    )
    if add_noise:
        image += sim_api.noise_for_model(model=image)
    return image


def sharp_image(
    lens_class, band, mag_zero_point, delta_pix, num_pix, with_deflector=True
):
    """Creates an unconvolved image of a selected lens. Point source image is not
    included in this function.

    :param lens_class: Lens() object
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :param with_deflector: bool, if True includes deflector light
    :return: 2d array unblurred image
    """
    kwargs_model, kwargs_params = lens_class.lenstronomy_kwargs(band)
    kwargs_band = {
        "pixel_scale": delta_pix,
        "magnitude_zero_point": mag_zero_point,
        "background_noise": 0,  # these are keywords not being used but need to be
        ## set in SimAPI
        "psf_type": "NONE",  # these are keywords not being used but need to be set
        ##in SimAPI
        "exposure_time": 1,
    }  # these are keywords not being used but need to be set in
    ##SimAPI
    sim_api = SimAPI(
        numpix=num_pix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model
    )
    kwargs_lens_light, kwargs_source, kwargs_ps = sim_api.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_params.get("kwargs_lens_light", None),
        kwargs_source_mag=kwargs_params.get("kwargs_source", None),
        kwargs_ps_mag=kwargs_params.get("kwargs_ps", None),
    )
    kwargs_numerics = {"supersampling_factor": 1}
    image_model = sim_api.image_model_class(kwargs_numerics)
    kwargs_lens = kwargs_params.get("kwargs_lens", None)
    image = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_ps=kwargs_ps,
        unconvolved=True,
        source_add=True,
        lens_light_add=with_deflector,
        point_source_add=False,
    )
    return image


def sharp_rgb_image(lens_class, rgb_band_list, mag_zero_point, delta_pix, num_pix):
    """Creates an unconvolved rgb image of a selected lens.

    :param lens_class: Lens() object
    :param rgb_band_list: imaging band list. Here, 'i', 'r', and 'g' band are consider
        as r, g, and b respectively.
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :return: rgb image
    """
    image_r = sharp_image(
        lens_class=lens_class,
        band=rgb_band_list[0],
        mag_zero_point=mag_zero_point,
        delta_pix=delta_pix,
        num_pix=num_pix,
    )
    image_g = sharp_image(
        lens_class=lens_class,
        band=rgb_band_list[1],
        mag_zero_point=mag_zero_point,
        delta_pix=delta_pix,
        num_pix=num_pix,
    )
    image_b = sharp_image(
        lens_class=lens_class,
        band=rgb_band_list[2],
        mag_zero_point=mag_zero_point,
        delta_pix=delta_pix,
        num_pix=num_pix,
    )
    image_rgb = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
    return image_rgb


def rgb_image_from_image_list(image_list, stretch):
    """Creates a rgb image using list of images in r, g, and b.

    :param image_list: images in r, g, and b band. Here, 'i', 'r', and 'g' band are
        consider as r, g, and b respectively.
    :return: rgb image
    """
    image_rgb = make_lupton_rgb(
        image_list[0], image_list[1], image_list[2], stretch=stretch
    )
    return image_rgb


def centered_coordinate_system(num_pix, transform_pix2angle):
    """Returns dictionary for Coordinate Grid such that (0,0) is centered with given
    input orientation coordinate transformation matrix.

    :param num_pix: number of pixels
    :type num_pix: int
    :param transform_pix2angle: transformation matrix (2x2) of pixels into coordinate
        displacements
    :return: dict with ra_at_xy_0, dec_at_xy_0, transfrom_pix2angle
    """
    pix_center = (num_pix - 1) / 2
    ra_center = (
        pix_center * transform_pix2angle[0, 0] + pix_center * transform_pix2angle[1, 0]
    )
    dec_center = (
        pix_center * transform_pix2angle[0, 1] + pix_center * transform_pix2angle[1, 1]
    )

    kwargs_grid = {
        "ra_at_xy_0": -ra_center,
        "dec_at_xy_0": -dec_center,
        "transform_pix2angle": transform_pix2angle,
    }
    return kwargs_grid


def image_data_class(
    lens_class, band, mag_zero_point, delta_pix, num_pix, transform_pix2angle
):
    kwargs_model, kwargs_params = lens_class.lenstronomy_kwargs(band)
    kwargs_band = {
        "pixel_scale": delta_pix,
        "magnitude_zero_point": mag_zero_point,
        "background_noise": 0,
        "psf_type": "NONE",
        "exposure_time": 1,
        "kwargs_pixel_grid": centered_coordinate_system(num_pix, transform_pix2angle),
    }
    sim_api = SimAPI(
        numpix=num_pix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model
    )

    imagedata = sim_api.data_class
    return imagedata


def point_source_coordinate_properties(
    lens_class, band, mag_zero_point, delta_pix, num_pix, transform_pix2angle
):
    """Provides pixel coordinates for deflector and images. Currently, this function
    only works for point source.

    :param lens_class: Lens() object
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :return: astropy table of deflector and image coordinate in pixel unit and other
        coordinate properties.
    """

    image_data = image_data_class(
        lens_class, band, mag_zero_point, delta_pix, num_pix, transform_pix2angle
    )

    lens_center = lens_class.deflector_position
    ra_lens_value = lens_center[0]
    dec_lens_value = lens_center[1]
    lens_pix_coordinate = image_data.map_coord2pix(ra_lens_value, dec_lens_value)

    ps_coordinate = lens_class.image_positions()
    ra_image_values = ps_coordinate[0]
    dec_image_values = ps_coordinate[1]
    # image_magnitude = lens_class.point_source_magnitude(band=band, lensed=True)
    image_pix_coordinate = []
    for image_ra, image_dec in zip(ra_image_values, dec_image_values):
        image_pix_coordinate.append((image_data.map_coord2pix(image_ra, image_dec)))

    data = Table(
        [
            lens_pix_coordinate,
            image_pix_coordinate,
            ra_image_values,
            dec_image_values,
        ],
        names=(
            "deflector_pix",
            "image_pix",
            "ra_image",
            "dec_image",
        ),
    )
    return data


def point_source_image_without_variability(
    lens_class,
    band,
    mag_zero_point,
    delta_pix,
    num_pix,
    psf_kernels,
    transform_pix2angle,
):
    """Creates lensed point source images without variability on the basis of given
    information.

    :param lens_class: Lens() object
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :param psf_kernels: psf kernels.
    :return: point source images
    """

    image_data = point_source_coordinate_properties(
        lens_class=lens_class,
        band=band,
        mag_zero_point=mag_zero_point,
        delta_pix=delta_pix,
        num_pix=num_pix,
        transform_pix2angle=transform_pix2angle,
    )

    data_class = image_data_class(
        lens_class, band, mag_zero_point, delta_pix, num_pix, transform_pix2angle
    )

    ra_image_values = image_data["ra_image"]
    dec_image_values = image_data["dec_image"]
    psf_class = []
    for i in range(len(psf_kernels)):
        psf_class.append(PSF(psf_type="PIXEL", kernel_point_source=psf_kernels[i]))

    magnitude = lens_class.point_source_magnitude(band, lensed=True)
    amp = magnitude_to_amplitude(magnitude, mag_zero_point)
    point_source_images = []
    for i in range(len(psf_class)):
        rendering_class = PointSourceRendering(
            pixel_grid=data_class, supersampling_factor=1, psf=psf_class[i]
        )
        point_source = rendering_class.point_source_rendering(
            np.array([ra_image_values[i]]),
            np.array([dec_image_values[i]]),
            np.array([amp[i]]),
        )
        point_source_images.append(point_source)
    return point_source_images


def point_source_image_at_time(
    lens_class,
    band,
    mag_zero_point,
    delta_pix,
    num_pix,
    psf_kernels,
    transform_pix2angle,
    time,
):
    """Creates lensed point source images with variability at a given time on the basis
    of given information.

    :param lens_class: Lens() object
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :param psf_kernels: psf kernels for the given exposure.
    :param time: time is a image observation time [day].
    :return: point source images with variability
    """

    image_data = point_source_coordinate_properties(
        lens_class=lens_class,
        band=band,
        mag_zero_point=mag_zero_point,
        delta_pix=delta_pix,
        num_pix=num_pix,
        transform_pix2angle=transform_pix2angle,
    )

    data_class = image_data_class(
        lens_class, band, mag_zero_point, delta_pix, num_pix, transform_pix2angle
    )

    ra_image_values = image_data["ra_image"]
    dec_image_values = image_data["dec_image"]
    psf_class = []
    for i in range(len(psf_kernels)):
        psf_class.append(PSF(psf_type="PIXEL", kernel_point_source=psf_kernels[i]))

    variable_mag = lens_class.point_source_magnitude(band=band, lensed=True, time=time)
    variable_amp = magnitude_to_amplitude(variable_mag, mag_zero_point)
    point_source_images = []
    for i in range(len(psf_class)):
        rendering_class = PointSourceRendering(
            pixel_grid=data_class, supersampling_factor=1, psf=psf_class[i]
        )
        point_source = rendering_class.point_source_rendering(
            np.array([ra_image_values[i]]),
            np.array([dec_image_values[i]]),
            variable_amp[i],
        )
        point_source_images.append(point_source)
    return np.array(point_source_images)


def point_source_image_with_variability(
    lens_class,
    band,
    mag_zero_point,
    delta_pix,
    num_pix,
    psf_kernels,
    transform_pix2angle,
    t_obs,
):
    """Creates lensed point source images with variability for series of time on the
    basis of given information.

    :param lens_class: Lens() object
    :param band: imaging band
    :param mag_zero_point: magnitude zero point for each exposure
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :param psf_kernels: psf kernels in the sequence of exposures being simulated.
    :param t_obs: array of image observation time [day].
    :return: array of point source images of each source with variability
    """
    all_image = []
    for time, psf_kernel, mag_zero, transf_matrix in zip(
        t_obs, psf_kernels, mag_zero_point, transform_pix2angle
    ):
        image_test = point_source_image_at_time(
            lens_class,
            band=band,
            mag_zero_point=mag_zero,
            delta_pix=delta_pix,
            num_pix=num_pix,
            psf_kernels=psf_kernel,
            transform_pix2angle=transf_matrix,
            time=time,
        )
        all_image.append(image_test)
    variab_images = [list(x) for x in zip(*all_image)]
    return variab_images
