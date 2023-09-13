import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import constants
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from sim_pipeline.ParamDistributions.gaussian_mixture_model import GaussianMixtureModel
from lenstronomy.Util import util, data_util


class GalaxyGalaxyLens(object):
    """Class to manage individual galaxy-galaxy lenses."""

    def __init__(
        self,
        source_dict,
        deflector_dict,
        cosmo,
        source_type="extended",
        test_area=4 * np.pi,
        mixgauss_means=None,
        mixgauss_stds=None,
        mixgauss_weights=None,
        magnification_limit=0.01,
    ):
        """

        :param source_dict: source properties
        :type source_dict: dict
        :param deflector_dict: deflector properties
        :type deflector_dict: dict
        :param cosmo: astropy.cosmology instance
        :param source_type: type of the source 'extended' or 'point_source' supported
        :type source_type: str
        :param test_area: area of disk around one lensing galaxies to be investigated
            on (in arc-seconds^2)
        :param mixgauss_weights: weights of the Gaussian mixture
        :param mixgauss_stds: standard deviations of the Gaussian mixture
        :param mixgauss_means: means of the Gaussian mixture
        :type mixgauss_weights: list of float
        :type mixgauss_stds: list of float
        :type mixgauss_means: list of float
        :param magnification_limit: absolute lensing magnification lower limit to
            register a point source (ignore highly de-magnified images)
        :type magnification_limit: float >= 0
        """
        self._source_dict = source_dict
        self._lens_dict = deflector_dict
        self.cosmo = cosmo
        self._source_type = source_type
        self.test_area = test_area
        self._mixgauss_means = mixgauss_means
        self._mixgauss_stds = mixgauss_stds
        self._mixgauss_weights = mixgauss_weights
        self._magnification_limit = magnification_limit
        if self._lens_dict["z"] >= self._source_dict["z"]:
            self._theta_E_sis = 0
        else:
            lens_cosmo = LensCosmo(
                z_lens=float(self._lens_dict["z"]),
                z_source=float(self._source_dict["z"]),
                cosmo=self.cosmo,
            )
            self._theta_E_sis = lens_cosmo.sis_sigma_v2theta_E(
                float(self._lens_dict["vel_disp"])
            )

    @property
    def lens_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        if not hasattr(self, "_center_lens"):
            center_x_lens, center_y_lens = np.random.normal(
                loc=0, scale=0.1
            ), np.random.normal(loc=0, scale=0.1)
            self._center_lens = np.array([center_x_lens, center_y_lens])
        return self._center_lens

    @property
    def source_position(self):
        """Source position, either the center of the extended source or the point
        source. If not present from the catalog, it is drawn uniformly within the circle
        of the test area centered on the lens.

        :return: [x_pos, y_pos]
        """
        center_lens = self.lens_position

        if not hasattr(self, "_center_source"):
            # Define the radius of the test area circle
            test_area_radius = np.sqrt(self.test_area / np.pi)
            # Randomly generate a radius within the test area circle
            r = np.sqrt(np.random.random()) * test_area_radius
            theta = 2 * np.pi * np.random.random()
            # Convert polar coordinates to cartesian coordinates
            center_x_source = center_lens[0] + r * np.cos(theta)
            center_y_source = center_lens[1] + r * np.sin(theta)
            self._center_source = np.array([center_x_source, center_y_source])
        return self._center_source

    def image_positions(self):
        """Return image positions by solving the lens equation. These are either the
        centers of the extended source, or the point sources in case of (added) point-
        like sources, such as quasars or SNe.

        :return: x-pos, y-pos
        """
        if not hasattr(self, "_image_positions"):
            lens_model_list, kwargs_lens = self.lens_model_lenstronomy()
            lens_model_class = LensModel(lens_model_list=lens_model_list)
            lens_eq_solver = LensEquationSolver(lens_model_class)
            source_pos_x, source_pos_y = self.source_position
            # TODO: analytical solver possible but currently does not support the
            #  convergence term
            self._image_positions = lens_eq_solver.image_position_from_source(
                source_pos_x,
                source_pos_y,
                kwargs_lens,
                solver="lenstronomy",
                search_window=self.einstein_radius * 6,
                min_distance=self.einstein_radius * 6 / 100,
                magnification_limit=self._magnification_limit,
            )
        return self._image_positions

    def validity_test(
        self, min_image_separation=0, max_image_separation=10, mag_arc_limit=None
    ):
        """Check whether lensing configuration matches selection and plausibility
        criteria.

        :param min_image_separation: minimum image separation
        :param max_image_separation: maximum image separation
        :param mag_arc_limit: dictionary with key of bands and values of magnitude
            limits of integrated lensed arc
        :type mag_arc_limit: dict with key of bands and values of magnitude limits
        :return: boolean
        """
        # Criteria 1:The redshift of the lens (z_lens) must be less than the redshift of
        # the source (z_source).
        z_lens = self._lens_dict["z"]
        z_source = self._source_dict["z"]
        if z_lens >= z_source:
            return False

        # Criteria 2: The angular Einstein radius of the lensing configuration (theta_E)
        # times 2 must be greater than or equal to the minimum image separation
        # (min_image_separation) and less than or equal to the maximum image
        # separation (max_image_separation).
        if not min_image_separation <= 2 * self._theta_E_sis <= max_image_separation:
            return False

        # Criteria 3: The distance between the lens center and the source position must
        # be less than or equal to the angular Einstein radius of the lensing
        # configuration (times sqrt(2)).
        center_lens, center_source = self.lens_position, self.source_position

        if np.sum((center_lens - center_source) ** 2) > self._theta_E_sis**2 * 2:
            return False

        # Criteria 4: The lensing configuration must produce at least two SL images.
        image_positions = self.image_positions()
        if len(image_positions[0]) < 2:
            return False

        # Criteria 5: The maximum separation between any two image positions must be
        # greater than or equal to the minimum image separation and less than or
        # equal to the maximum image separation.
        image_separation = image_separation_from_positions(image_positions)
        if not min_image_separation <= image_separation <= max_image_separation:
            return False

        # Criteria 6: (optional)
        # compute the magnified brightness of the lensed extended arc for different
        # bands at least in one band, the magnitude has to be brighter than the limit
        if mag_arc_limit is not None and self._source_type in ["extended"]:
            # makes sure magnification of extended source is only used when there is
            # an extended source
            bool_mag_limit = False
            host_mag = self.host_magnification()
            for band, mag_limit_band in mag_arc_limit.items():
                mag_source = self.extended_source_magnitude(band)
                mag_arc = mag_source - 2.5 * np.log10(
                    host_mag
                )  # lensing magnification results in a shift in magnitude
                if mag_arc < mag_limit_band:
                    bool_mag_limit = True
                    break
            if bool_mag_limit is False:
                return False
        # TODO make similar criteria for point source magnitudes
        return True
        # TODO: test for signal-to-noise ratio in surface brightness

    @property
    def lens_redshift(self):
        """

        :return: lens redshift
        """
        return self._lens_dict["z"]

    @property
    def source_redshift(self):
        """

        :return: source redshift
        """
        return self._source_dict["z"]

    @property
    def einstein_radius(self):
        """Einstein radius, including the SIS + external convergence effect.

        :return: Einstein radius [arc seconds]
        """
        _, _, kappa_ext = self.los_linear_distortions()
        return self._theta_E_sis / (1 - kappa_ext)

    def deflector_ellipticity(self):
        """

        :return: e1_light, e2_light, e1_mass, e2_mass
        """
        e1_light, e2_light = float(self._lens_dict["e1_light"]), float(
            self._lens_dict["e2_light"]
        )
        e1_mass, e2_mass = float(self._lens_dict["e1_mass"]), float(
            self._lens_dict["e2_mass"]
        )
        return e1_light, e2_light, e1_mass, e2_mass

    def deflector_stellar_mass(self):
        """

        :return: stellar mass of deflector
        """
        return self._lens_dict["stellar_mass"]

    def deflector_velocity_dispersion(self):
        """

        :return: velocity dispersion [km/s]
        """
        return self._lens_dict["vel_disp"]

    def los_linear_distortions(self):
        """Line-of-sight distortions in shear and convergence.

        :return: kappa, gamma1, gamma2
        """
        # TODO: more realistic distribution of shear and convergence,
        #  the covariances among them and redshift correlations
        mixgauss_means = self._mixgauss_means
        mixgauss_stds = self._mixgauss_stds
        mixgauss_weights = self._mixgauss_weights
        if not hasattr(self, "_gamma"):
            mixture = GaussianMixtureModel(
                means=mixgauss_means,
                stds=mixgauss_stds,
                weights=mixgauss_weights,
            )
            gamma = np.abs(mixture.rvs(size=1))[0]
            phi = 2 * np.pi * np.random.random()
            gamma1 = gamma * np.cos(2 * phi)
            gamma2 = gamma * np.sin(2 * phi)
            self._gamma = [gamma1, gamma2]
        if not hasattr(self, "_kappa"):
            self._kappa = np.random.normal(loc=0, scale=0.05)
        return self._gamma[0], self._gamma[1], self._kappa

    def deflector_magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        band_string = str("mag_" + band)
        return self._lens_dict[band_string]

    def point_source_arrival_times(self):
        """Arrival time of images relative to a straight line without lensing. Negative
        values correspond to images arriving earlier, and positive signs correspond to
        images arriving later.

        :return: arrival times for each image [days]
        :rtype: numpy array
        """
        lens_model_list, kwargs_lens = self.lens_model_lenstronomy()
        lens_model = LensModel(
            lens_model_list=lens_model_list,
            cosmo=self.cosmo,
            z_lens=self.lens_redshift,
            z_source=self.source_redshift,
        )
        x_image, y_image = self.image_positions()
        arrival_times = lens_model.arrival_time(
            x_image, y_image, kwargs_lens=kwargs_lens
        )
        return arrival_times

    def image_observer_times(self, t_obs):
        """Calculates time of the source at the different images, not correcting for
        redshifts, but for time delays. The time is relative to the first arriving
        image.

        :param t_obs: time of observation [days]
        :return: time of the source when seen in the different images (without redshift
            correction)
        :rtype: numpy array
        """
        arrival_times = self.point_source_arrival_times()
        observer_times = t_obs + arrival_times - np.min(arrival_times)
        return observer_times

    def point_source_magnitude(self, band, lensed=False):
        """Point source magnitude, either unlensed (single value) or lensed (array) with
        macro-model magnifications.

        # TODO: time-variability with time-delayed and micro-lensing

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: point source magnitude
        """
        band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        source_mag = self._source_dict[band_string]
        # TODO: requires time information and needs to be shifted for
        # different arriving images
        if lensed:
            mag = self.point_source_magnification()
            return source_mag - 2.5 * np.log10(np.abs(mag))
        return source_mag

    def extended_source_magnitude(self, band, lensed=False):
        """Unlensed apparent magnitude of the extended source for a given band (assumes
        that size is the same for different bands)

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        source_mag = self._source_dict[band_string]
        if lensed:
            mag = self.host_magnification()
            return source_mag - 2.5 * np.log10(mag)
        return source_mag

    def point_source_magnification(self):
        """Macro-model magnification of point sources.

        :return: signed magnification of point sources in same order as image positions
        """
        if not hasattr(self, "_ps_magnification"):
            lens_model_list, kwargs_lens = self.lens_model_lenstronomy()
            lensModel = LensModel(lens_model_list=lens_model_list)
            img_x, img_y = self.image_positions()
            self._ps_magnification = lensModel.magnification(img_x, img_y, kwargs_lens)
        return self._ps_magnification

    def host_magnification(self):
        """Compute the extended lensed surface brightness and calculates the integrated
        flux-weighted magnification factor of the extended host galaxy.

        :return: integrated magnification factor of host magnitude
        """
        if not hasattr(self, "_host_magnification"):
            kwargs_model, kwargs_params = self.lenstronomy_kwargs(band=None)
            lightModel = LightModel(
                light_model_list=kwargs_model.get("source_light_model_list", [])
            )
            lensModel = LensModel(
                lens_model_list=kwargs_model.get("lens_model_list", [])
            )
            theta_E = self.einstein_radius
            center_source = self.source_position

            kwargs_source_mag = kwargs_params["kwargs_source"]
            kwargs_source_amp = data_util.magnitude2amplitude(
                lightModel, kwargs_source_mag, magnitude_zero_point=0
            )

            num_pix = 200
            delta_pix = theta_E * 4 / num_pix
            x, y = util.make_grid(numPix=200, deltapix=delta_pix)
            x += center_source[0]
            y += center_source[1]
            beta_x, beta_y = lensModel.ray_shooting(x, y, kwargs_params["kwargs_lens"])
            flux_lensed = np.sum(
                lightModel.surface_brightness(beta_x, beta_y, kwargs_source_amp)
            )
            flux_no_lens = np.sum(
                lightModel.surface_brightness(x, y, kwargs_source_amp)
            )
            if flux_no_lens > 0:
                self._host_magnification = flux_lensed / flux_no_lens
            else:
                self._host_magnification = 0
        return self._host_magnification

    def lenstronomy_kwargs(self, band=None):
        """

        :param band: imaging band, if =None, will result in un-normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions

        """
        lens_model_list, kwargs_lens = self.lens_model_lenstronomy()
        kwargs_model = {
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "lens_model_list": lens_model_list,
        }

        center_lens, center_source = self.lens_position, self.source_position
        if self._source_type == "extended":
            # convert radian to arc seconds
            if band is None:
                mag_source = 1
            else:
                mag_source = self.extended_source_magnitude(band)
            size_source_arcsec = (
                float(self._source_dict["angular_size"]) / constants.arcsec
            )
            kwargs_model["source_light_model_list"] = ["SERSIC_ELLIPSE"]
            kwargs_source = [
                {
                    "magnitude": mag_source,
                    "R_sersic": size_source_arcsec,
                    "n_sersic": float(self._source_dict["n_sersic"]),
                    "e1": float(self._source_dict["e1"]),
                    "e2": float(self._source_dict["e2"]),
                    "center_x": center_source[0],
                    "center_y": center_source[1],
                }
            ]
        else:
            kwargs_source = None

        if self._source_type == "point_source":
            kwargs_model["point_source_model"] = ["LENSED_POSITION"]
            img_x, img_y = self.image_positions()
            if band is None:
                image_magnitudes = np.abs(self.point_source_magnification())
            else:
                image_magnitudes = self.point_source_magnitude(band=band, lensed=True)
            kwargs_ps = [
                {"ra_image": img_x, "dec_image": img_y, "point_amp": image_magnitudes}
            ]
        else:
            kwargs_ps = None

        e1_light_lens, e2_light_lens, e1_mass, e2_mass = self.deflector_ellipticity()
        size_lens_arcsec = (
            self._lens_dict["angular_size"] / constants.arcsec
        )  # convert radian to arc seconds

        if band is None:
            mag_lens = 1
        else:
            mag_lens = self.deflector_magnitude(band)
        kwargs_lens_light = [
            {
                "magnitude": mag_lens,
                "R_sersic": size_lens_arcsec,
                "n_sersic": float(self._source_dict["n_sersic"]),
                "e1": e1_light_lens,
                "e2": e2_light_lens,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            }
        ]

        kwargs_params = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }
        return kwargs_model, kwargs_params

    def lens_model_lenstronomy(self):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :return: lens_model_list, kwargs_lens
        """
        lens_model_list = ["EPL", "SHEAR", "CONVERGENCE"]
        theta_E = self.einstein_radius
        e1_light_lens, e2_light_lens, e1_mass, e2_mass = self.deflector_ellipticity()
        center_lens = self.lens_position
        gamma1, gamma2, kappa_ext = self.los_linear_distortions()
        kwargs_lens = [
            {
                "theta_E": theta_E,
                "gamma": 2,
                "e1": e1_mass,
                "e2": e2_mass,
                "center_x": center_lens[0],
                "center_y": center_lens[1],
            },
            {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0, "dec_0": 0},
            {"kappa": kappa_ext, "ra_0": 0, "dec_0": 0},
        ]
        return lens_model_list, kwargs_lens


def image_separation_from_positions(image_positions):
    """Calculate image separation in arc-seconds; if there are only two images, the
    separation between them is returned; if there are more than 2 images, the maximum
    separation is returned.

    :param image_positions: list of image positions in arc-seconds
    :return: image separation in arc-seconds
    """
    if len(image_positions[0]) == 2:
        image_separation = np.sqrt(
            (image_positions[0][0] - image_positions[0][1]) ** 2
            + (image_positions[1][0] - image_positions[1][1]) ** 2
        )
    else:
        coords = np.stack((image_positions[0], image_positions[1]), axis=-1)
        separations = np.sqrt(
            np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=-1)
        )
        image_separation = np.max(separations)
    return image_separation


def theta_e_when_source_infinity(deflector_dict=None, v_sigma=None):
    """Calculate Einstein radius in arc-seconds for a source at infinity.

    :param deflector_dict: deflector properties
    :param v_sigma: velocity dispersion in km/s
    :return: Einstein radius in arc-seconds
    """
    if v_sigma is None:
        if deflector_dict is None:
            raise ValueError("Either deflector_dict or v_sigma must be provided")
        else:
            v_sigma = deflector_dict["vel_disp"]

    theta_E_infinity = (
        4 * np.pi * (v_sigma * 1000.0 / constants.c) ** 2 / constants.arcsec
    )
    return theta_E_infinity
