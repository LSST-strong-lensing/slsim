from lenstronomy.SimulationAPI.sim_api import SimAPI


def simulate_image(lens_class, band, num_pix, add_noise=True, observatory='LSST', **kwargs):
    """

    :param lens_class: class object containing all information of the lensing system (e.g., GGLens())
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
    kwargs_single_band = image_quality_lenstronomy.kwargs_single_band(observatory=observatory, band=band, **kwargs)

    sim_api = SimAPI(numpix=num_pix, kwargs_single_band=kwargs_single_band, kwargs_model=kwargs_model)
    kwargs_lens_light, kwargs_source, kwargs_ps = sim_api.magnitude2amplitude(kwargs_lens_light_mag=kwargs_params.get('kwargs_lens_light', None),
                                                                              kwargs_source_mag=kwargs_params.get('kwargs_source', None),
                                                                              kwargs_ps_mag=kwargs_params.get('kwargs_ps', None))
    kwargs_numerics = {'point_source_supersampling_factor': 1, 'supersampling_factor': 3}
    image_model = sim_api.image_model_class(kwargs_numerics)
    kwargs_lens = kwargs_params.get('kwargs_lens', None)
    image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light,
                              kwargs_ps=kwargs_ps)
    if add_noise:
        image += sim_api.noise_for_model(model=image)
    return image
