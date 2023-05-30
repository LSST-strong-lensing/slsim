import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util import util, data_util


def image_separation_from_positions(image_positions):
    """
    calculate image separation in arc-seconds; if there are only two images, the separation between them is returned;
    if there are more than 2 images, the maximum separation is returned

    :param image_positions: list of image positions in arc-seconds
    :return: image separation in arc-seconds
    """
    if len(image_positions[0]) == 2:
        image_separation = np.sqrt((image_positions[0][0] - image_positions[0][1]) ** 2 + (
                image_positions[1][0] - image_positions[1][1]) ** 2)
    else:
        coords = np.stack((image_positions[0], image_positions[1]), axis=-1)
        separations = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=-1))
        image_separation = np.max(separations)
    return image_separation


def theta_e_when_source_infinity(deflector_dict=None, v_sigma=None):
    """
            calculate Einstein radius in arc-seconds for a source at infinity

            :param deflector_dict: deflector properties
            :param v_sigma: velocity dispersion in km/s
            :return: Einstein radius in arc-seconds
            """
    if v_sigma is None:
        if deflector_dict is None:
            raise ValueError("Either deflector_dict or v_sigma must be provided")
        else:
            v_sigma = deflector_dict['vel_disp']

    theta_E_infinity = 4 * np.pi * (v_sigma * 1000. / constants.c) ** 2 / constants.arcsec
    return theta_E_infinity


class GGLens(object):
    """
    class to manage individual galaxy-galaxy lenses
    """

    def __init__(self, source_dict, deflector_dict, cosmo, test_area=4 * np.pi):
        """

        :param source_dict: source properties
        :type source_dict: dict
        :param deflector_dict: deflector properties
        :type deflector_dict: dict
        :param cosmo: astropy.cosmology instance
        :param test_area: area of disk around one lensing galaxies to be investigated on (in arc-seconds^2)
        """
        self._source_dict = source_dict
        self._lens_dict = deflector_dict
        self.cosmo = cosmo
        self.test_area = test_area
        if self._lens_dict['z'] >= self._source_dict['z']:
            self._theta_E = 0
        else:
            lens_cosmo = LensCosmo(z_lens=float(self._lens_dict['z']), z_source=float(self._source_dict['z']),
                                   cosmo=self.cosmo)
            self._theta_E = lens_cosmo.sis_sigma_v2theta_E(float(self._lens_dict['vel_disp']))

    def position_alignment(self):
        """
        Draws position of the lens and source in arcseconds.lens and source center positions as 2D lists

        :return: [center_x_lens, center_y_lens], [center_x_source, center_y_source] in arc-seconds

        """
        if not hasattr(self, '_center_lens'):
            center_x_lens, center_y_lens = np.random.normal(loc=0, scale=0.1), np.random.normal(loc=0, scale=0.1)
            self._center_lens = np.array([center_x_lens, center_y_lens])
        # TODO: make it more realistic scatter

        if not hasattr(self, '_center_source'):
            # Define the radius of the test area circle
            test_area_radius = np.sqrt(self.test_area / np.pi)
            # Randomly generate a radius within the test area circle
            r = np.sqrt(np.random.random()) * test_area_radius
            theta = 2 * np.pi * np.random.random()
            # Convert polar coordinates to cartesian coordinates
            center_x_source = self._center_lens[0] + r * np.cos(theta)
            center_y_source = self._center_lens[1] + r * np.sin(theta)
            self._center_source = np.array([center_x_source, center_y_source])
        return self._center_lens, self._center_source

    def get_image_positions(self):
        if not hasattr(self, '_image_positions'):
            kwargs_model, kwargs_params = self.lenstronomy_kwargs('g')
            lens_model_list = kwargs_model['lens_model_list']
            lens_model_class = LensModel(lens_model_list=lens_model_list)
            lens_eq_solver = LensEquationSolver(lens_model_class)
            source_pos_x = kwargs_params['kwargs_source'][0]['center_x']
            source_pos_y = kwargs_params['kwargs_source'][0]['center_y']

            kwargs_lens = kwargs_params['kwargs_lens']
            # TODO: analytical solver possible but currently does not support the convergence term
            self._image_positions = lens_eq_solver.image_position_from_source(source_pos_x, source_pos_y, kwargs_lens,
                                                                              solver='lenstronomy')
        return self._image_positions

    def validity_test(self, min_image_separation=0, max_image_separation=10, mag_arc_limit=None):
        """
        check whether lensing configuration matches selection and plausibility criteria

        :param min_image_separation: minimum image separation
        :param max_image_separation: maximum image separation
        :param mag_arc_limit: dictionary with key of bands and values of magnitude limits of integrated lensed arc
        :type mag_arc_limit: dict with key of bands and values of magnitude limits
        :return: boolean
        """
        # Criteria 1:The redshift of the lens (z_lens) must be less than the redshift of the source (z_source).
        z_lens = self._lens_dict['z']
        z_source = self._source_dict['z']
        if z_lens >= z_source:
            return False

        # Criteria 2: The angular Einstein radius of the lensing configuration (theta_E) times 2 must be greater than
        # or equal to the minimum image separation (min_image_separation) and less than or equal to the maximum image
        # separation (max_image_separation).
        if self._theta_E * 2 < min_image_separation or self._theta_E * 2 > max_image_separation:
            return False

        # Criteria 3: The distance between the lens center and the source position must be less than or equal to the
        # angular Einstein radius of the lensing configuration (times sqrt(2)).
        center_lens, center_source = self.position_alignment()
        if np.sum((center_lens - center_source)**2) > self._theta_E ** 2 * 2:
            return False

        # Criteria 4: The lensing configuration must produce at least two SL images.
        image_positions = self.get_image_positions()
        if len(image_positions[0]) < 2:
            return False

        # Criteria 5: The maximum separation between any two image positions must be greater than or equal to the
        # minimum image separation and less than or equal to the maximum image separation.
        image_separation = image_separation_from_positions(image_positions)
        if image_separation < min_image_separation:
            return False
        if image_separation > max_image_separation:
            return False

        # Criteria 6: (optional)
        # compute the magnified brightness of the lensed extended arc for different bands
        # at least in one band, the magnitude has to be brighter than the limit
        if mag_arc_limit is not None:
            bool_mag_limit = False
            host_mag = self.host_magnification()
            for band, mag_limit_band in mag_arc_limit.items():
                mag_source = self.source_magnitude(band)
                mag_arc = mag_source - 2.5 * np.log10(host_mag)  # lensing magnification results in a shift in magnitude
                if mag_arc < mag_limit_band:
                    bool_mag_limit = True
                    break
            if bool_mag_limit is False:
                return False
        return True
        # TODO: test for SN ratio in surface brightness

    @property
    def lens_redshift(self):
        """

        :return: lens redshift
        """
        return self._lens_dict['z']

    @property
    def source_redshift(self):
        """

        :return: source redshift
        """
        return self._source_dict['z']

    @property
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
        e1_light, e2_light = float(self._lens_dict['e1_light']), float(self._lens_dict['e2_light'])
        e1_mass, e2_mass = float(self._lens_dict['e1_mass']), float(self._lens_dict['e2_mass'])
        return e1_light, e2_light, e1_mass, e2_mass

    def deflector_stellar_mass(self):
        """

        :return: stellar mass of deflector
        """
        return self._lens_dict['stellar_mass']

    def deflector_velocity_dispersion(self):
        """

        :return: velocity dispersion [km/s]
        """
        return self._lens_dict['vel_disp']

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

    def host_magnification(self):
        """
        compute the extended lensed surface brightness and calculates the integrated flux-weighted magnification factor
        of the extended host galaxy

        :return: integrated magnification factor of host magnitude
        """
        kwargs_model, kwargs_params = self.lenstronomy_kwargs(band=None)

        lightModel = LightModel(light_model_list=kwargs_model.get('source_light_model_list', []))
        lensModel = LensModel(lens_model_list=kwargs_model.get('lens_model_list', []))
        theta_E = self.einstein_radius
        center_lens, center_source = self.position_alignment()

        kwargs_source_mag = kwargs_params['kwargs_source']
        kwargs_source_amp = data_util.magnitude2amplitude(lightModel, kwargs_source_mag, magnitude_zero_point=0)

        num_pix = 200
        delta_pix = theta_E * 4 / num_pix
        x, y = util.make_grid(numPix=200, deltapix=delta_pix)
        x += center_source[0]
        y += center_source[1]
        beta_x, beta_y = lensModel.ray_shooting(x, y, kwargs_params['kwargs_lens'])
        flux_lensed = np.sum(lightModel.surface_brightness(beta_x, beta_y, kwargs_source_amp))
        flux_no_lens = np.sum(lightModel.surface_brightness(x, y, kwargs_source_amp))
        if flux_no_lens > 0:
            return flux_lensed / flux_no_lens
        return None

    def lenstronomy_kwargs(self, band=None):
        """

        :param band: imaging band, if =None, will result in un-normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        lens_model_list, kwargs_lens = self.lens_model_lenstronomy()
        kwargs_model = {'source_light_model_list': ['SERSIC_ELLIPSE'],
                        'lens_light_model_list': ['SERSIC_ELLIPSE'],
                        'lens_model_list':  lens_model_list}

        e1_light_lens, e2_light_lens, e1_mass, e2_mass = self.deflector_ellipticity()
        center_lens, center_source = self.position_alignment()

        size_lens_arcsec = self._lens_dict['angular_size'] / constants.arcsec  # convert radian to arc seconds
        if band is None:
            mag_lens = 1
            mag_source = 1
        else:
            mag_lens = self.deflector_magnitude(band)
            mag_source = self.source_magnitude(band)
        kwargs_lens_light = [{'magnitude': mag_lens, 'R_sersic': size_lens_arcsec,
                              'n_sersic': float(self._source_dict['n_sersic']),
                              'e1': e1_light_lens, 'e2': e2_light_lens,
                              'center_x': center_lens[0], 'center_y': center_lens[1]}]

        size_source_arcsec = float(self._source_dict['angular_size']) / constants.arcsec  # convert radian to arc seconds

        kwargs_source = [{'magnitude': mag_source, 'R_sersic': size_source_arcsec,
                          'n_sersic': float(self._source_dict['n_sersic']),
                          'e1': float(self._source_dict['e1']), 'e2': float(self._source_dict['e2']),
                          'center_x': center_source[0], 'center_y': center_source[1]}]

        kwargs_params = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source,
                         'kwargs_lens_light': kwargs_lens_light}
        return kwargs_model, kwargs_params

    def lens_model_lenstronomy(self):
        """
        returns lens model instance and parameters in lenstronomy conventions

        :return: lens_model_list, kwargs_lens
        """
        lens_model_list = ['EPL', 'SHEAR', 'CONVERGENCE']
        theta_E = self.einstein_radius
        e1_light_lens, e2_light_lens, e1_mass, e2_mass = self.deflector_ellipticity()
        center_lens, center_source = self.position_alignment()
        gamma1, gamma2, kappa_ext = self.los_linear_distortions()
        kwargs_lens = [{'theta_E': theta_E, 'gamma': 2, 'e1': e1_mass, 'e2': e2_mass,
                        'center_x': center_lens[0], 'center_y': center_lens[1]},
                       {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0, 'dec_0': 0},
                       {'kappa': kappa_ext, 'ra_0': 0, 'dec_0': 0}]
        return lens_model_list, kwargs_lens