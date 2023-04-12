import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants


class GGLens(object):
    """
    class to manage individual galaxy-galaxy lenses
    """

    def __init__(self, source_dict, deflector_dict, cosmo):
        """

        :param source_dict: source properties
        :type source_dict: dict
        :param deflector_dict: deflector properties
        :type deflector_dict: dict
        :param cosmo: astropy.cosmology instance
        """
        self._source_dict = source_dict
        self._lens_dict = deflector_dict
        self.cosmo = cosmo

        self._lens_cosmo = LensCosmo(z_lens=self._lens_dict['z'], z_source=self._source_dict['z'], cosmo=self.cosmo)
        self._theta_E = self._lens_cosmo.sis_sigma_v2theta_E(self._lens_dict['vel_disp'])

    def position_alignment(self):
        """
        draws position of the lens and source

        :return:
        """
        if not hasattr(self, '_center_lens'):
            center_x_lens, center_y_lens = np.random.normal(loc=0, scale=0.1), np.random.normal(loc=0, scale=0.1)
            self._center_lens = [center_x_lens, center_y_lens]
        if not hasattr(self, '_center_source'):
            r_squared, theta = [self.einstein_radius() * np.sqrt(np.random.random()), 2*np.pi*np.random.random()]
            center_x_source = np.sqrt(r_squared) * np.cos(theta)
            center_y_source = np.sqrt(r_squared) * np.sin(theta)
            self._center_source = [center_x_source, center_y_source]
        return self._center_lens, self._center_source

    def validity_test(self, min_image_separation=0, max_image_separation=10):
        """
        check whether lensing configuration matches selection and plausibility criteria

        :param min_image_separation:
        :param max_image_separation:
        :return: boolean
        """
        z_lens = self._lens_dict['z']
        z_source = self._source_dict['z']
        if z_lens >= z_source:
            return False
        if self._theta_E * 2 < min_image_separation:
            return False
        if self._theta_E * 2 > max_image_separation:
            return False
        # TODO: test for image multiplicities
        # TODO: test for lensed arc brightness
        return True

    def einstein_radius(self):
        """
        Einstein radius

        :return: Einstein radius [arc seconds]
        """
        return self._theta_E

    def deflector_ellipticity(self):
        """

        :return: e1_light, e2_light, e1_mass, e2_mass
        """
        e1_light, e2_light = self._lens_dict['e1_light'], self._lens_dict['e2_light']
        e1_mass, e2_mass = self._lens_dict['e1_mass'], self._lens_dict['e2_mass']
        return e1_light, e2_light, e1_mass, e2_mass

    def los_linear_distortions(self):
        """
        line-of-sight distortions in shear and convergence

        :return: kappa, gamma1, gamma2
        """
        # TODO: more realistic distribution of shear and convergence,
        #  the covariances among them and redshift correlations
        if not hasattr(self, '_gamma'):

            gamma = np.random.normal(loc=0, scale=0.1)
            phi = 2 * np.pi * np.random.random()
            gamma1 = gamma * np.cos(2 * phi)
            gamma2 = gamma * np.sin(2 * phi)
            self._gamma = [gamma1, gamma2]
        if not hasattr(self, '_kappa'):
            self._kappa = np.random.normal(loc=0, scale=0.05)
        return self._gamma[0], self._gamma[1], self._kappa

    def deflector_magnitude(self, band):
        """
        apparent magnitude of the deflector for a given band

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        band_string = str('mag_' + band)
        return self._lens_dict[band_string]

    def source_magnitude(self, band):
        """
        unlensed apparent magnitude of the source for a given band

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        band_string = str('mag_' + band)
        return self._source_dict[band_string]

    def lenstronomy_kwargs(self, band):
        """

        :param band: imaging band
        :type band: string
        :return: lenstronomy model and parameter conventions
        """
        kwargs_model = {'source_light_model_list': ['SERSIC_ELLIPSE'],
                        'lens_light_model_list': ['SERSIC_ELLIPSE'],
                        'lens_model_list': ['EPL', 'SHEAR', 'CONVERGENCE']}
        theta_E = self.einstein_radius()
        e1_light_lens, e2_light_lens, e1_mass, e2_mass = self.deflector_ellipticity()
        center_lens, center_source = self.position_alignment()
        gamma1, gamma2, kappa_ext = self.los_linear_distortions()
        kwargs_lens = [{'theta_E': theta_E, 'gamma': 2, 'e1': e1_mass, 'e2': e2_mass,
                        'center_x': center_lens[0], 'center_y': center_lens[1]},
                       {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0, 'dec_0': 0},
                       {'kappa': kappa_ext, 'ra_0': 0, 'dec_0': 0}]
        size_lens_arcsec = self._lens_dict['angular_size'] / constants.arcsec  # convert radian to arc seconds
        mag_lens = self.deflector_magnitude(band)
        kwargs_lens_light = [{'magnitude': mag_lens, 'R_sersic': size_lens_arcsec, 'n_sersic': self._lens_dict['n_sersic'],
                              'e1': e1_light_lens, 'e2': e2_light_lens,
                              'center_x': center_lens[0], 'center_y': center_lens[1]}]

        size_source_arcsec = self._source_dict['angular_size'] / constants.arcsec  # convert radian to arc seconds
        mag_source = self.source_magnitude(band)
        kwargs_source = [{'magnitude': mag_source, 'R_sersic': size_source_arcsec,
                          'n_sersic': self._source_dict['n_sersic'],
                          'e1': self._source_dict['e1'], 'e2': self._source_dict['e2'],
                          'center_x': center_source[0], 'center_y': center_source[1]}]

        kwargs_params = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source,
                         'kwargs_lens_light': kwargs_lens_light}
        return kwargs_model, kwargs_params
