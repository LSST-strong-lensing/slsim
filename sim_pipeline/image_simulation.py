import numpy as np
from astropy.table import Table
from lenstronomy.SimulationAPI.sim_api import SimAPI
from astropy.visualization import make_lupton_rgb


def simulate_image(
    lens_class, band, num_pix, add_noise=True, observatory="LSST", **kwargs
):
    """Creates an image of a selected lens with noise.

    :param lens_class: class object containing all information of the lensing system
        (e.g., GalaxyGalaxyLens())
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
    from sim_pipeline.Observations import image_quality_lenstronomy

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
    """Creates a high resolution image of a selected lens.

    :param lens_class: GalaxyGalaxyLens() object
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
        point_source_add=False
    )
    return image


def sharp_rgb_image(lens_class, rgb_band_list, mag_zero_point, delta_pix, num_pix):
    """Creates a high resolution rgb image of a selected lens.

    :param lens_class: GalaxyGalaxyLens() object
    :param rgb_band_list: imaging band list
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

    :param image_list: images in r, g, and b band
    :return: rgb image
    """
    image_rgb = make_lupton_rgb(
        image_list[0], image_list[1], image_list[2], stretch=stretch
    )
    return image_rgb

def point_source_image_properties(
    lens_class, band, mag_zero_point, delta_pix, num_pix):
    """provides pixel coordinates for deflector and images. Currently, 
    this function only works for point source. 

    :param lens_class: GalaxyGalaxyLens() object
    :param band: imaging band
    :param mag_zero_point: magnitude zero point in band
    :param delta_pix: pixel scale of image generated
    :param num_pix: number of pixels per axis
    :return: astropy table of deflector and image coordinate in pixel unit and other properties
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

    image_data = sim_api.data_class
    
    ra_lens_value = kwargs_params['kwargs_lens'][1]['ra_0']
    dec_lens_value = kwargs_params['kwargs_lens'][1]['dec_0']
    lens_pix_coordinate = image_data.map_coord2pix(ra_lens_value, dec_lens_value)

    ps_coordinate = kwargs_params['kwargs_ps']
    ra_image_values = [item['ra_image'] for item in ps_coordinate]
    dec_image_values = [item['dec_image'] for item in ps_coordinate]
    image_magnitude = [item['magnitude'] for item in ps_coordinate]
    image_pix_coordinate = []
    for image_ra, image_dec in zip(ra_image_values[0], dec_image_values[0]):
        image_pix_coordinate.append((image_data.map_coord2pix(image_ra, image_dec)))
    ra_at_xy_0, dec_at_xy_0 = image_data.map_pix2coord(0, 0)
    
    kwargs_lens_light, kwargs_source, kwargs_ps = sim_api.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_params.get("kwargs_lens_light", None),
        kwargs_source_mag=kwargs_params.get("kwargs_source", None),
        kwargs_ps_mag=kwargs_params.get("kwargs_ps", None),
    )
    image_amplitude = [item['point_amp'] for item in kwargs_ps]

    """data = {'deflector_pix': (lens_pix_coordinate[0], lens_pix_coordinate[1]), 
            'image_pix': image_pix_coordinate, 'image_amplitude': image_amplitude[0], 'image_magnitude': image_magnitude[0],
            'radec_at_xy_0': np.array([ra_at_xy_0, dec_at_xy_0])}"""
    data = Table(
        [
            (lens_pix_coordinate[0], lens_pix_coordinate[1]),
            image_pix_coordinate, ra_image_values[0], dec_image_values[0],
            image_amplitude[0],
            image_magnitude[0],
            np.array([ra_at_xy_0, dec_at_xy_0]),
        ],
        names=(
            "deflector_pix",
            "image_pix", "ra_image", "dec_image",
            "image_amplitude",
            "image_magnitude",
            "radec_at_xy_0",
        ),
    )
    return data
