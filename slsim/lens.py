import warnings

import numpy as np
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.lens_equation_solver import (
    analytical_lens_model_support,
)
from slsim.ParamDistributions.los_config import LOSConfig
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import constants
from lenstronomy.Util import data_util
from lenstronomy.Util import util

from slsim.lensed_system_base import LensedSystemBase


class Lens(LensedSystemBase):
    """Class to manage individual lenses."""

    def __init__(
        self,
        source_class,
        deflector_class,
        cosmo,
        lens_equation_solver="lenstronomy_analytical",
        test_area=4 * np.pi,
        magnification_limit=0.01,
        los_config=None,
        los_dict=None,
    ):
        """

        :param source_class: A Source class instance or list of Source class instance
        :type source_class: Source class instance from slsim.Sources.source. Eg: 
         source_class=Source(
            source_dict=source_dict,
            variability_model=variability_model,
            kwargs_variability=kwargs_variability,
            sn_type=sn_type,
            sn_absolute_mag_band=sn_absolute_mag_band,
            sn_absolute_zpsys=sn_absolute_zpsys,
            cosmo=cosmo,
            lightcurve_time=lightcurve_time,
            sn_modeldir=sn_modeldir,
            agn_driving_variability_model=agn_driving_variability_model,
            agn_driving_kwargs_variability=agn_driving_kwargs_variability,
            source_type=source_type,
            light_profile=light_profile,
        ). See the Source class documentation.
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
         Eg: deflector_class = Deflector(
            deflector_type=deflector_type,
            deflector_dict=deflector_dict,
        ). See the Deflector class documentation.
        :param cosmo: astropy.cosmology instance
        :param lens_equation_solver: type of lens equation solver; currently supporting
         "lenstronomy_analytical" and "lenstronomy_general"
        :type lens_equation_solver: str
        :param test_area: solid angle around one lensing galaxies to be investigated
            on (in arc-seconds^2)
        :param magnification_limit: absolute lensing magnification lower limit to
            register a point source (ignore highly de-magnified images)
        :type magnification_limit: float >= 0
        :param los_config: LOSConfig instance which manages line-of-sight (LOS) effects
         and Gaussian mixture models in a simulation or analysis context.
        :param los_dict: line of sight dictionary (optional, takes these values instead of drawing from distribution)
         Takes "gamma" = [gamma1, gamma2] and "kappa" = kappa as entries
        :type los_dict: dict
        """
        self.deflector = deflector_class
        self.source = source_class
        self.cosmo = cosmo
        self.test_area = test_area
        self._source_type = self.source.source_type
        self._lens_equation_solver = lens_equation_solver
        self._magnification_limit = magnification_limit

        if isinstance(self.source, list):
            self.single_source_class = max(self.source, key=lambda obj: obj.redshift)
            #source_z = self.single_source_class.redshift
            self._source_type = self.single_source_class.source_type
            self.source_number = len(self.source)
        else:
            #source_z = self.source.redshift
            self._source_type = self.source.source_type
            self.source_number = 1
            self.single_source_class = self.source
        # we conventionally use highest source redshift in the lens cosmo.
        self._lens_cosmo = LensCosmo(
            z_lens=float(self.deflector.redshift),
            z_source=float(self.single_source_class.redshift),
            cosmo=self.cosmo,
        )

        self._los_linear_distortions_cache = None
        self.los_config = los_config
        if self.los_config is None:
            if los_dict is None:
                los_dict = {}
            self.los_config = LOSConfig(**los_dict)

    @property
    def image_number(self):
        """Number of images in the lensing configuration.

        :return: number of images
        """
        return len(self.point_source_image_positions()[0])

    @property
    def deflector_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        return self.deflector.deflector_center

    def extended_source_image_positions(self):
        """Returns extended source image positions by solving the lens equation.

        :return: x-pos, y-pos
        """
        if not hasattr(self, "_image_positions"):
            lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
            lens_model_class = LensModel(lens_model_list=lens_model_list)
            lens_eq_solver = LensEquationSolver(lens_model_class)
            source_pos_x, source_pos_y = self.source.extended_source_position(
                center_lens=self.deflector_position, draw_area=self.test_area
            )
            if (
                self._lens_equation_solver == "lenstronomy_analytical"
                and analytical_lens_model_support(lens_model_list) is True
            ):
                solver = "analytical"
            else:
                solver = "lenstronomy"
            self._image_positions = lens_eq_solver.image_position_from_source(
                source_pos_x,
                source_pos_y,
                kwargs_lens,
                solver=solver,
                search_window=self.einstein_radius * 6,
                min_distance=self.einstein_radius * 6 / 200,
                magnification_limit=self._magnification_limit,
            )
        return self._image_positions

    def point_source_image_positions(self):
        """Returns point source image positions by solving the lens equation. In the
        absence of a point source, this function returns the solution for the center of
        the extended source.

        :return: x-pos, y-pos
        """
        if not hasattr(self, "_point_image_positions"):
            lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
            lens_model_class = LensModel(lens_model_list=lens_model_list)
            lens_eq_solver = LensEquationSolver(lens_model_class)
            point_source_pos_x, point_source_pos_y = self.source.point_source_position(
                center_lens=self.deflector_position, draw_area=self.test_area
            )
            # uses analytical lens equation solver in case it is supported by lenstronomy for speed-up
            if (
                self._lens_equation_solver == "lenstronomy_analytical"
                and analytical_lens_model_support(lens_model_list) is True
            ):
                solver = "analytical"
            else:
                solver = "lenstronomy"
            self._point_image_positions = lens_eq_solver.image_position_from_source(
                point_source_pos_x,
                point_source_pos_y,
                kwargs_lens,
                solver=solver,
                search_window=self.einstein_radius * 6,
                min_distance=self.einstein_radius * 6 / 200,
                magnification_limit=self._magnification_limit,
            )
        return self._point_image_positions

    def validity_test(
        self,
        min_image_separation=0,
        max_image_separation=10,
        mag_arc_limit=None,
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
        # Criteria 1:The redshift of the lens (z_lens) must be less than the
        # redshift of the source (z_source).
        z_lens = self.deflector.redshift
        z_source = self.source.redshift
        if z_lens >= z_source:
            return False

        # Criteria 2: The angular Einstein radius of the lensing configuration (theta_E)
        # times 2 must be greater than or equal to the minimum image separation
        # (min_image_separation) and less than or equal to the maximum image
        # separation (max_image_separation).
        if not min_image_separation <= 2 * self.einstein_radius <= max_image_separation:
            return False

        # Criteria 3: The distance between the lens center and the source position
        # must be less than or equal to the angular Einstein radius
        # of the lensing configuration (times sqrt(2)).
        center_lens, center_source = (
            self.deflector_position,
            self.source.point_source_position(
                center_lens=self.deflector_position, draw_area=self.test_area
            ),
        )
        if np.sum((center_lens - center_source) ** 2) > self.einstein_radius**2 * 2:
            return False

        # Criteria 4: The lensing configuration must produce at least two SL images.
        image_positions = self.point_source_image_positions()
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
        if mag_arc_limit is not None and self._source_type in [
            "extended",
            "point_plus_extended",
        ]:
            # makes sure magnification of extended source is only used when there is
            # an extended source
            bool_mag_limit = False
            host_mag = self.extended_source_magnification()
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
    def deflector_redshift(self):
        """

        :return: lens redshift
        """
        return self.deflector.redshift

    @property
    def source_redshift(self):
        """

        :return: source redshift
        """
        return self.source.redshift

    @property
    def external_convergence(self):
        """

        :return: external convergence
        """
        _, _, kappa_ext = self.los_linear_distortions
        return kappa_ext

    @property
    def external_shear(self):
        """

        :return: the absolute external shear
        """
        gamma1, gamma2, _ = self.los_linear_distortions
        return (gamma1**2 + gamma2**2) ** 0.5

    @property
    def einstein_radius_deflector(self):
        """Einstein radius, from SIS approximation (coming from velocity dispersion)
        without line-of-sight correction.

        :return:
        """
        if not hasattr(self, "_theta_E"):
            if self.deflector.redshift >= self.source.redshift:
                self._theta_E = 0
            elif self.deflector.deflector_type in ["EPL"]:
                self._theta_E = self._lens_cosmo.sis_sigma_v2theta_E(
                    float(self.deflector.velocity_dispersion(cosmo=self.cosmo))
                )
            else:
                # numerical solution for the Einstein radius
                lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
                lens_model = LensModel(lens_model_list=lens_model_list)
                lens_analysis = LensProfileAnalysis(lens_model=lens_model)
                self._theta_E = lens_analysis.effective_einstein_radius(
                    kwargs_lens, r_min=1e-3, r_max=5e1, num_points=50
                )
        return self._theta_E

    @property
    def einstein_radius(self):
        """Einstein radius, from SIS approximation (coming from velocity dispersion) +
        external convergence effect.

        :return: Einstein radius [arc seconds]
        """
        theta_E = self.einstein_radius_deflector
        _, _, kappa_ext = self.los_linear_distortions
        return theta_E / (1 - kappa_ext)

    def deflector_ellipticity(self):
        """

        :return: e1_light, e2_light, e1_mass, e2_mass
        """
        e1_light, e2_light = self.deflector.light_ellipticity
        e1_mass, e2_mass = self.deflector.mass_ellipticity
        return e1_light, e2_light, e1_mass, e2_mass

    def deflector_stellar_mass(self):
        """

        :return: stellar mass of deflector
        """
        return self.deflector.stellar_mass

    def deflector_velocity_dispersion(self):
        """

        :return: velocity dispersion [km/s]
        """
        return self.deflector.velocity_dispersion(cosmo=self.cosmo)

    @property
    def los_linear_distortions(self):
        if self._los_linear_distortions_cache is None:
            self._los_linear_distortions_cache = (
                self._calculate_los_linear_distortions()
            )
        return self._los_linear_distortions_cache

    def _calculate_los_linear_distortions(self):
        """Line-of-sight distortions in shear and convergence.

        :return: kappa, gamma1, gamma2
        """
        return self.los_config.calculate_los_linear_distortions(
            source_redshift=self.source_redshift,
            deflector_redshift=self.deflector_redshift,
        )

    def deflector_magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        return self.deflector.magnitude(band=band)

    def point_source_arrival_times(self):
        """Arrival time of images relative to a straight line without lensing. Negative
        values correspond to images arriving earlier, and positive signs correspond to
        images arriving later.

        :return: arrival times for each image [days]
        :rtype: numpy array
        """
        lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        lens_model = LensModel(
            lens_model_list=lens_model_list,
            cosmo=self.cosmo,
            z_lens=self.deflector_redshift,
            z_source=self.source_redshift,
        )
        x_image, y_image = self.point_source_image_positions()
        arrival_times = lens_model.arrival_time(
            x_image, y_image, kwargs_lens=kwargs_lens
        )
        return arrival_times

    def image_observer_times(self, t_obs):
        """Calculates time of the source at the different images, not correcting for
        redshifts, but for time delays. The time is relative to the first arriving
        image.

        :param t_obs: time of observation [days]. It could be a single observation time
            or an array of observation time.
        :return: time of the source when seen in the different images (without redshift
            correction)
        :rtype: numpy array. Each element of the array corresponds to different image
            observation times.
        """
        arrival_times = self.point_source_arrival_times()
        if type(t_obs) is np.ndarray and len(t_obs) > 1:
            observer_times = (
                t_obs[:, np.newaxis] - arrival_times + np.min(arrival_times)
            ).T
        else:
            observer_times = (t_obs - arrival_times + np.min(arrival_times))[
                :, np.newaxis
            ]

        return observer_times

    def point_source_magnitude(self, band, lensed=False, time=None):
        """Point source magnitude, either unlensed (single value) or lensed (array) with
        macro-model magnifications.

        # TODO: time-variability with micro-lensing

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :param time: time is a image observation time in units of days. If None,
            provides magnitude without variability.
        :return: point source magnitude
        """
        # TODO: might have to change conventions between extended and point source
        if lensed:
            magnif = self.point_source_magnification()
            magnif_log = 2.5 * np.log10(abs(magnif))
            if time is not None:
                time = time
                image_observed_times = self.image_observer_times(time)
                variable_magnitude = self.source.point_source_magnitude(
                    band,
                    image_observation_times=image_observed_times,
                )
                lensed_variable_magnitude = (
                    variable_magnitude - magnif_log[:, np.newaxis]
                )
                return lensed_variable_magnitude
            else:
                source_mag_unlensed = self.source.point_source_magnitude(band)
                magnified_mag_list = []
                for i in range(len(magnif_log)):
                    magnified_mag_list.append(source_mag_unlensed - magnif_log[i])
                return np.array(magnified_mag_list)
        return self.source.point_source_magnitude(band)

    def extended_source_magnitude(self, band, lensed=False):
        """Unlensed apparent magnitude of the extended source for a given band (assumes
        that size is the same for different bands)

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        # band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        source_mag = self.source.extended_source_magnitude(band)
        if lensed:
            mag = self.extended_source_magnification()
            return source_mag - 2.5 * np.log10(mag)
        return source_mag

    def point_source_magnification(self):
        """Macro-model magnification of point sources.

        :return: signed magnification of point sources in same order as image positions
        """
        if not hasattr(self, "_ps_magnification"):
            lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
            lensModel = LensModel(lens_model_list=lens_model_list)
            img_x, img_y = self.point_source_image_positions()
            self._ps_magnification = lensModel.magnification(img_x, img_y, kwargs_lens)
        return self._ps_magnification

    def extended_source_magnification(self):
        """Compute the extended lensed surface brightness and calculates the integrated
        flux-weighted magnification factor of the extended host galaxy.

        :return: integrated magnification factor of host magnitude
        """
        if not hasattr(self, "_extended_source_magnification"):
            kwargs_model, kwargs_params = self.lenstronomy_kwargs(band=None)
            lightModel = LightModel(
                light_model_list=kwargs_model.get("source_light_model_list", [])
            )
            lensModel = LensModel(
                lens_model_list=kwargs_model.get("lens_model_list", [])
            )
            theta_E = self.einstein_radius
            center_source = self.source.extended_source_position(
                center_lens=self.deflector_position, draw_area=self.test_area
            )

            kwargs_source_mag = kwargs_params["kwargs_source"]
            kwargs_source_amp = data_util.magnitude2amplitude(
                lightModel, kwargs_source_mag, magnitude_zero_point=0
            )

            num_pix = 200
            delta_pix = theta_E * 4 / num_pix
            x, y = util.make_grid(numPix=num_pix, deltapix=delta_pix)
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
                self._extended_source_magnification = flux_lensed / flux_no_lens
            else:
                self._extended_source_magnification = 0
        return self._extended_source_magnification

    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        lens_mass_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        (
            lens_light_model_list,
            kwargs_lens_light,
        ) = self.deflector.light_model_lenstronomy(band=band)
        # list of 
        kwargs_model = {
            "lens_light_model_list": lens_light_model_list,
            "lens_model_list": lens_mass_model_list,
            # source_redshift_list: redshifts for each source_light component
            # z_source_convention: source redshift for convention of lens parameters (make sure LensCosmo has the same one)
            # cosmo: astropy cosmology
            # point_source_redshift_list:
        }

        sources, sources_kwargs = self.source_light_model_lenstronomy(band=band)
        # ensure that only the models that exist are getting added to kwargs_model
        for k in sources.keys():
            kwargs_model[k] = sources[k]

        kwargs_source = sources_kwargs["kwargs_source"]
        kwargs_ps = sources_kwargs["kwargs_ps"]

        kwargs_params = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }

        return kwargs_model, kwargs_params

    def deflector_mass_model_lenstronomy(self):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :return: lens_model_list, kwargs_lens
        """
        if self.deflector.deflector_type in ["EPL", "NFW_HERNQUIST", "NFW_CLUSTER"]:
            lens_mass_model_list, kwargs_lens = self.deflector.mass_model_lenstronomy(
                lens_cosmo=self._lens_cosmo
            )
        else:
            raise ValueError(
                "Deflector model %s not supported for lenstronomy model"
                % self.deflector.deflector_type
            )
        # adding line-of-sight structure
        gamma1, gamma2, kappa_ext = self.los_linear_distortions
        gamma1_lenstronomy, gamma2_lenstronomy = ellipticity_slsim_to_lenstronomy(
            e1_slsim=gamma1, e2_slsim=gamma2
        )
        kwargs_lens.append(
            {
                "gamma1": gamma1_lenstronomy,
                "gamma2": gamma2_lenstronomy,
                "ra_0": 0,
                "dec_0": 0,
            }
        )
        kwargs_lens.append({"kappa": kappa_ext, "ra_0": 0, "dec_0": 0})
        lens_mass_model_list.append("SHEAR")
        lens_mass_model_list.append("CONVERGENCE")

        return lens_mass_model_list, kwargs_lens

    def deflector_light_model_lenstronomy(self, band):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        return self.deflector.light_model_lenstronomy(band=band)

    def source_light_model_lenstronomy(self, band=None):
        """Returns source light model instance and parameters in lenstronomy
        conventions.

        :return: source_light_model_list, kwargs_source_light
        """
        source_models = {}
        all_source_kwarg_dict = {}
        if (
            self._source_type == "extended"
            or self._source_type == "point_plus_extended"
        ):
            source_models["source_light_model_list"
                          ] = self.source.extended_source_light_model()
            kwargs_source = self.source.kwargs_extended_source_light(
                draw_area=self.test_area, center_lens=self.deflector_position, band=band
            )
        else:
            # source_models['source_light_model_list'] = None
            kwargs_source = None

        if (
            self._source_type == "point_source"
            or self._source_type == "point_plus_extended"
        ):
            source_models["point_source_model_list"] = ["LENSED_POSITION"]
            img_x, img_y = self.point_source_image_positions()
            if band is None:
                image_magnitudes = np.abs(self.point_source_magnification())
            else:
                image_magnitudes = self.point_source_magnitude(band=band, lensed=True)
            kwargs_ps = [
                {"ra_image": img_x, "dec_image": img_y, "magnitude": image_magnitudes}
            ]
        else:
            # source_models['point_source_model'] = None
            kwargs_ps = None
        all_source_kwarg_dict["kwargs_source"] = kwargs_source
        all_source_kwarg_dict["kwargs_ps"] = kwargs_ps
        return source_models, all_source_kwarg_dict

    def kappa_star(self, ra, dec):
        """Computes the stellar surface density at location (ra, dec) in units of
        lensing convergence.

        :param ra: position in the image plane
        :param dec: position in the image plane
        :return: kappa_star
        """
        stellar_mass = self.deflector_stellar_mass()
        kwargs_model, kwargs_params = self.lenstronomy_kwargs(band=None)
        lightModel = LightModel(
            light_model_list=kwargs_model.get("lens_light_model_list", [])
        )
        kwargs_lens_light_mag = kwargs_params["kwargs_lens_light"]
        kwargs_lens_light_amp = data_util.magnitude2amplitude(
            lightModel, kwargs_lens_light_mag, magnitude_zero_point=0
        )

        total_flux = lightModel.total_flux(kwargs_lens_light_amp)  # integrated flux
        flux_local = lightModel.surface_brightness(
            ra, dec, kwargs_lens_light_amp
        )  # surface brightness per arcsecond square
        kappa_star = (
            flux_local / total_flux * stellar_mass / self._lens_cosmo.sigma_crit_angle
        )
        return kappa_star


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
