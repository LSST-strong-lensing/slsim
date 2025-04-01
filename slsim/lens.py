import copy

import numpy as np
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.lens_equation_solver import (
    analytical_lens_model_support,
)

from slsim.Util.param_util import (ellipticity_slsim_to_lenstronomy, 
                                   source_position)
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
        los_class=None,
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
        :param los_class: line of sight dictionary (optional, takes these values instead of drawing from distribution)
        :type los_class: ~LOSIndividual() class object
        """
        super().__init__(
            source_class=source_class,
            deflector_class=deflector_class,
            los_class=los_class,
        )
        self.cosmo = cosmo
        self.test_area = test_area
        self._lens_equation_solver = lens_equation_solver
        self._magnification_limit = magnification_limit

        if isinstance(source_class, list):
            self._source = source_class
            # chose a highest resdshift source to use conventionally use in lens
            #  mass model.
            self.max_redshift_source_class = max(
                self._source, key=lambda obj: obj.redshift
            )
            self.source_number = len(self._source)
            self._max_redshift_source_index = self._source.index(
                self.max_redshift_source_class
            )
        else:
            self._source = [source_class]
            self.source_number = 1
            # this is for single source case. self.max_redshift_source_class and
            # self.source are the same class. The difference is only that one is in the
            #  form of list and other is just a Source instance. This is done just for
            # the completion of routine to make things consistent in both single source
            # and double source case.
            self.max_redshift_source_class = source_class
            self._max_redshift_source_index = 0
        self._source_type = self.max_redshift_source_class.source_type
        # we conventionally use highest source redshift in the lens cosmo.
        self._lens_cosmo = LensCosmo(
            z_lens=self.deflector.redshift,
            z_source=self.max_redshift_source_class.redshift,
            cosmo=self.cosmo,
        )

    def source(self, index=0):
        """

        :param index: index of the source
        :type index: int
        :return: Source() class with index
        """
        return self._source[index]

    @property
    def image_number(self):
        """Number of images in the lensing configuration.

        :return: number of images
        """
        return [len(pos[0]) for pos in self.point_source_image_positions()]

    @property
    def deflector_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        return self.deflector.deflector_center
    
    def extended_source_postion_list(self):
        """Provides list of extended source center"""
        if not hasattr(self, "_source_pos_list"):
            self._source_pos_list = []
            for source in self._source:
                single_source_pos = self._extended_source_position(source)
                self._source_pos_list.append(single_source_pos)
        return self._source_pos_list
    
    def extended_source_postion_list(self):
        """Provides list of point source center"""
        if not hasattr(self, "_point_source_pos_list"):
            self._point_source_pos_list = []
            for source in self._source:
                single_source_pos = self._extended_source_position(source)
                self._point_source_pos_list.append(single_source_pos)
        return self._point_source_pos_list

    def _extended_source_position(self, source):
        """Extended source position for a single source. If the source already has a defined 
        extended source position, it is used. Otherwise, a new position is randomly drawn.

        :param source: The source object containing extended source properties.
        :return: (x_pos, y_pos) coordinates of the extended source.
        """
        source_pos_x, source_pos_y = source.extended_source_position

        if source_pos_x is None:
            source_pos_x, source_pos_y = source_position(
                center_lens=self.deflector_position,
                draw_area=self.test_area
            )
        if not hasattr(self, "_center_source"):
            self._center_source = np.array([source_pos_x, source_pos_y])
        return self._center_source
    
    def _point_source_position(self, source):
        """Point source position for a single. The point source could be at the center of the
        extended source or offset from it. If a point source offset is not provided,
        it defaults to the extended source center.

        :param source: The source object containing point source properties.
        :return: [x_pos, y_pos]
        """

        # Get the extended source center
        extended_source_center = self._extended_source_position(source)

        # Apply offsets if provided
        if source.point_source_offset[0] is not None:
            center_x_point_source = extended_source_center[0] + source.point_source_offset[0]
            center_y_point_source = extended_source_center[1] + source.point_source_offset[1]
        else:
            center_x_point_source, center_y_point_source = extended_source_center
        return np.array([center_x_point_source, center_y_point_source])

    def extended_source_image_positions(self):
        """Returns extended source image positions by solving the lens equation
        for each source.

        :return: list of (x-pos, y-pos)
        """
        if not hasattr(self, "_es_image_position_list"):
            self._es_image_position_list = []
            for source in self._source:
                self._es_image_position_list.append(
                    self._extended_source_image_positions(source)
                )
        return self._es_image_position_list

    def _extended_source_image_positions(self, source):
        """Returns extended source image positions by solving the lens equation
        for a single source.

        :param source: Source class instance. The redshift of this
            source is used in the LensModel.
        :return: x-pos, y-pos
        """

        lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        lens_model_class = LensModel(
            lens_model_list=lens_model_list,
            z_lens=self.deflector_redshift,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
            z_source=source.redshift,
        )
        lens_eq_solver = LensEquationSolver(lens_model_class)
        source_pos_x, source_pos_y = self._extended_source_position(source=source)
        if (
            self._lens_equation_solver == "lenstronomy_analytical"
            and analytical_lens_model_support(lens_model_list) is True
        ):
            solver = "analytical"
        else:
            solver = "lenstronomy"
        einstein_radius = self._einstein_radius(source)
        self._image_positions = lens_eq_solver.image_position_from_source(
            source_pos_x,
            source_pos_y,
            kwargs_lens,
            solver=solver,
            search_window=einstein_radius * 6,
            min_distance=einstein_radius * 6 / 200,
            magnification_limit=self._magnification_limit,
        )
        return self._image_positions

    def point_source_image_positions(self):
        """Returns point source image positions by solving the lens equation
        for all sources. In the absence of a point source, this function
        returns the solution for the center of the extended source.

        :return: list of (x-pos, y-pos) for each source
        """
        if not hasattr(self, "_ps_image_position_list"):
            self._ps_image_position_list = []
            for source in self._source:
                self._ps_image_position_list.append(
                    self._point_source_image_positions(source)
                )
        return self._ps_image_position_list

    def _point_source_image_positions(self, source):
        """Returns point source image positions by solving the lens equation
        for a single source. In the absence of a point source, this function
        returns the solution for the center of the extended source.

        :param source: Source class instance. The redshift of this
            source is used in the LensModel.
        :return: x-pos, y-pos
        """
        lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        lens_model_class = LensModel(
            lens_model_list=lens_model_list,
            z_lens=self.deflector_redshift,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
            z_source=source.redshift,
        )
        lens_eq_solver = LensEquationSolver(lens_model_class)
        point_source_pos_x, point_source_pos_y = self._point_source_position(source)
        # uses analytical lens equation solver in case it is supported by lenstronomy for speed-up
        if (
            self._lens_equation_solver == "lenstronomy_analytical"
            and analytical_lens_model_support(lens_model_list) is True
        ):
            solver = "analytical"
        else:
            solver = "lenstronomy"
        einstein_radius = self._einstein_radius(source)
        self._point_image_positions = lens_eq_solver.image_position_from_source(
            point_source_pos_x,
            point_source_pos_y,
            kwargs_lens,
            solver=solver,
            search_window=einstein_radius * 6,
            min_distance=einstein_radius * 6 / 200,
            magnification_limit=self._magnification_limit,
        )
        return self._point_image_positions

    def validity_test(
        self,
        min_image_separation=0,
        max_image_separation=10,
        mag_arc_limit=None,
        second_brightest_image_cut=None,
    ):
        """Check whether multiple lensing configuration matches selection and
        plausibility criteria.

        :param min_image_separation: minimum image separation
        :param max_image_separation: maximum image separation
        :param mag_arc_limit: dictionary with key of bands and values of
            magnitude limits of integrated lensed arc
        :type mag_arc_limit: dict with key of bands and values of
            magnitude limits
        :param second_brightest_image_cut: Dictionary containing maximum
            magnitude of the second brightest image and corresponding
            band. If provided, selects lenses where the second brightest
            image has a magnitude less than or equal to provided
            magnitude. eg: second_bright_image_cut = {"band": "i",
            "second_bright_mag_max": 23}
        :return: A boolean or dict of boolean.
        """
        validity_results = {}
        for index, source in enumerate(self._source):
            validity_results[index] = self._validity_test(
                source,
                min_image_separation=min_image_separation,
                max_image_separation=max_image_separation,
                mag_arc_limit=mag_arc_limit,
                second_brightest_image_cut=second_brightest_image_cut,
                source_index=index,
            )
        if len(validity_results) == 1:
            return validity_results[0]
        else:
            return validity_results

    def _validity_test(
        self,
        source,
        min_image_separation=0,
        max_image_separation=10,
        mag_arc_limit=None,
        second_brightest_image_cut=None,
        source_index=None,
    ):
        """Check whether a single lensing configuration matches selection and
        plausibility criteria.

        :param min_image_separation: minimum image separation
        :param max_image_separation: maximum image separation
        :param mag_arc_limit: dictionary with key of bands and values of
            magnitude limits of integrated lensed arc
        :type mag_arc_limit: dict with key of bands and values of
            magnitude limits
        :param second_bright_image_cut: Dictionary containing maximum
            magnitude of the second brightest image and corresponding
            band. If provided, selects lenses where the second brightest
            image has a magnitude less than or equal to provided
            magnitude. eg: second_bright_image_cut = {"band": "i",
            "mag_max": 23}
        :param source_index: index of a source in source list.
        :return: boolean
        """

        # Criteria 1:The redshift of the lens (z_lens) must be less than the
        # redshift of the source (z_source).
        z_lens = self.deflector.redshift
        z_source = source.redshift
        if z_lens >= z_source:
            return False

        # Criteria 2: The angular Einstein radius of the lensing configuration (theta_E)
        # times 2 must be greater than or equal to the minimum image separation
        # (min_image_separation) and less than or equal to the maximum image
        # separation (max_image_separation).
        if (
            not min_image_separation
            <= 2 * self._einstein_radius(source)
            <= max_image_separation
        ):
            return False

        # Criteria 3: The distance between the lens center and the source position
        # must be less than or equal to the angular Einstein radius
        # of the lensing configuration (times sqrt(2)).
        if source.source_type in ["extended"]:
            source_pos = self._extended_source_position(source)
        elif source.source_type in ["point_source", "point_plus_extended"]:
            source_pos = self._point_source_position(source)
        center_lens, center_source = (self.deflector_position, source_pos)
        if (
            np.sum((center_lens - center_source) ** 2)
            > self._einstein_radius(source) ** 2 * 2
        ):
            return False

        # Criteria 4: The lensing configuration must produce at least two SL images.
        image_positions = self._point_source_image_positions(source)
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
        if mag_arc_limit is not None and source.source_type in [
            "extended",
            "point_plus_extended",
        ]:
            # makes sure magnification of extended source is only used when there is
            # an extended source
            bool_mag_limit = False
            host_mag = self._extended_single_source_magnification(source, source_index)
            for band, mag_limit_band in mag_arc_limit.items():
                mag_source = self._extended_source_magnitude(band, source, source_index)
                mag_arc = mag_source - 2.5 * np.log10(
                    host_mag
                )  # lensing magnification results in a shift in magnitude
                if mag_arc < mag_limit_band:
                    bool_mag_limit = True
                    break
            if bool_mag_limit is False:
                return False
        # TODO make similar criteria for point source magnitudes
        # Criteria 7: (optional)
        # computes the magnitude of each image and if the second brightest image has
        # the magnitude less or equal to "second_bright_mag_max" provided in the dict
        # second_bright_image_cut.
        if second_brightest_image_cut is not None:
            for band_max, mag_max in second_brightest_image_cut.items():
                if self._source_type == "extended":
                    image_magnitude_list = (
                        self.extended_source_magnitude_for_each_image(
                            band=band_max, lensed=True
                        )
                    )
                elif self._source_type in ["point_plus_extended", "point_source"]:
                    image_magnitude_list = self.point_source_magnitude(
                        band=band_max, lensed=True
                    )
                second_brightest_mag = np.sort(image_magnitude_list[0])[1]
                if second_brightest_mag > mag_max:
                    return False
        return True
        # TODO: test for signal-to-noise ratio in surface brightness

    @property
    def deflector_redshift(self):
        """

        :return: lens redshift
        """
        return self.deflector.redshift

    @property
    def source_redshift_list(self):
        """

        :return: list of source redshifts
        """
        source_redshifts = []
        for source in self._source:
            source_redshifts.append(source.redshift)
        return source_redshifts

    @property
    def los_linear_distortions(self):
        """Line-of-sight distortions in shear and convergence.

        :return: kappa, gamma1, gamma2
        """
        kappa = self.los_class.convergence
        gamma1, gamma2 = self.los_class.shear
        return kappa, gamma1, gamma2

    @property
    def external_convergence(self):
        """

        :return: external convergence
        """
        return self.los_class.convergence

    @property
    def external_shear(self):
        """

        :return: the absolute external shear
        """
        gamma1, gamma2 = self.los_class.shear
        return (gamma1**2 + gamma2**2) ** 0.5

    @property
    def einstein_radius(self):
        """Einstein radius, from SIS approximation (coming from velocity
        dispersion) without line-of-sight correction.

        :return: list of einstein radius of each lens-source pair.
        """
        if not hasattr(self, "_theta_E_list"):
            self._theta_E_list = []
            for source in self._source:
                self._theta_E_list.append(self._einstein_radius(source))
        return self._theta_E_list

    def _einstein_radius(self, source):
        """Einstein radius, including external shear.

        :param source: Source class instance. The redshift of this
            source is used in the LensCosmo or LensModel.
        :return: einstein radius of a lens-source pair.
        """
        if self.deflector.redshift >= source.redshift:
            theta_E = 0
            return theta_E
        lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        lens_model = LensModel(
            lens_model_list=lens_model_list,
            z_lens=self.deflector_redshift,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
            z_source=source.redshift,
        )
        if self.deflector.deflector_type in ["EPL"]:
            kappa_ext_convention = self.los_class.convergence
            gamma_pl = self.deflector.halo_properties
            theta_E_convention = kwargs_lens[0]["theta_E"]
            if source.redshift == self.max_redshift_source_class.redshift:
                theta_E = theta_E_convention
                kappa_ext = kappa_ext_convention
            else:
                beta = self._lens_cosmo.beta_double_source_plane(
                    z_lens=self.deflector_redshift,
                    z_source_2=self.max_redshift_source_class.redshift,
                    z_source_1=source.redshift,
                )
                theta_E = theta_E_convention * beta ** (1.0 / (gamma_pl - 1))
                kappa_ext = kappa_ext_convention * beta

            theta_E /= (1 - kappa_ext) ** (1.0 / (gamma_pl - 1))
        else:
            # numerical solution for the Einstein radius
            lens_analysis = LensProfileAnalysis(lens_model=lens_model)
            kwargs_lens_ = copy.deepcopy(kwargs_lens)
            kwargs_lens_[0]["center_x"] = 0
            kwargs_lens_[0]["center_y"] = 0
            theta_E = lens_analysis.effective_einstein_radius(
                kwargs_lens_, r_min=1e-4, r_max=5e1, num_points=100
            )
        return theta_E

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

    def deflector_magnitude(self, band):
        """Apparent magnitude of the deflector for a given band.

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        return self.deflector.magnitude(band=band)

    def point_source_arrival_times(self):
        """Arrival time of images relative to a straight line without lensing.
        Negative values correspond to images arriving earlier, and positive
        signs correspond to images arriving later. This is for single source.

        :return: list of arrival times for each image [days] for each
            source.
        :rtype: list of numpy array
        """
        arrival_times_list = []
        for source in self._source:
            arrival_times_list.append(self._point_source_arrival_times(source))
        return arrival_times_list

    def _point_source_arrival_times(self, source):
        """Arrival time of images relative to a straight line without lensing.
        Negative values correspond to images arriving earlier, and positive
        signs correspond to images arriving later.

        :return: arrival times for each image [days]
        :rtype: numpy array
        """
        lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        lens_model = LensModel(
            lens_model_list=lens_model_list,
            cosmo=self.cosmo,
            z_lens=self.deflector_redshift,
            z_source=source.redshift,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
        )
        x_image, y_image = self._point_source_image_positions(source)
        arrival_times = lens_model.arrival_time(
            x_image, y_image, kwargs_lens=kwargs_lens
        )
        return arrival_times

    def image_observer_times(self, t_obs):
        """Calculates time of the source at the different images, not
        correcting for redshifts, but for time delays. The time is relative to
        the first arriving image.

        :param t_obs: time of observation [days]. It could be a single
            observation time or an array of observation time.
        :return: time of the source when seen in the different images
            (without redshift correction)
        :rtype: list of numpy array. Each element of the array
            corresponds to different image observation times.
        """
        observer_times_list = []
        for source in self._source:
            observer_times_list.append(self._image_observer_times(source, t_obs))
        if self.source_number == 1:
            return observer_times_list[0]
        return observer_times_list

    def _image_observer_times(self, source, t_obs):
        """Calculates time of a source at the different images, not correcting
        for redshifts, but for time delays. The time is relative to the first
        arriving image.

        :param t_obs: time of observation [days]. It could be a single
            observation time or an array of observation time.
        :return: time of the source when seen in the different images
            (without redshift correction)
        :rtype: numpy array. Each element of the array corresponds to
            different image observation times.
        """
        arrival_times = self._point_source_arrival_times(source)
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
        """Point source magnitudes, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function provided
        magnitudes of all the sources.

        # TODO: time-variability with micro-lensing

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :param time: time is a image observation time in units of days.
            If None, provides magnitude without variability.
        :return: list of point source magnitudes.
        """

        magnitude_list = []
        for source in self._source:
            magnitude_list.append(
                self._point_source_magnitude(band, source, lensed=lensed, time=time)
            )
        return magnitude_list

    def _point_source_magnitude(self, band, source, lensed=False, time=None):
        """Point source magnitude, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function does operation
        only for the single source.

        # TODO: time-variability with micro-lensing

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :param time: time is a image observation time in units of days.
            If None, provides magnitude without variability.
        :return: point source magnitude of a single source
        """
        # TODO: might have to change conventions between extended and point source
        if lensed:
            magnif = self._point_source_magnification(source)
            magnif_log = 2.5 * np.log10(abs(magnif))
            if time is not None:
                time = time
                image_observed_times = self._image_observer_times(source, time)
                variable_magnitude = source.point_source_magnitude(
                    band,
                    image_observation_times=image_observed_times,
                )
                lensed_variable_magnitude = (
                    variable_magnitude - magnif_log[:, np.newaxis]
                )
                return lensed_variable_magnitude
            else:
                source_mag_unlensed = source.point_source_magnitude(band)
                magnified_mag_list = []
                for i in range(len(magnif_log)):
                    magnified_mag_list.append(source_mag_unlensed - magnif_log[i])
                return np.array(magnified_mag_list)
        return source.point_source_magnitude(band)

    def extended_source_magnitude_for_each_image(self, band, lensed=False):
        """Extended source magnitudes, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function provided
        magnitudes of all the sources. This function assumes that all the light
        of an extended source is concentrated at its center and magnifies it as
        a point source multiple times. For a more accurate lensed extended
        source magnitude, please see the extended_source_magnitude() function.

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
            of each images.
        :type lensed: bool
        :return: list of extended source magnitudes.
        """

        magnitude_list = []
        for source in self._source:
            magnitude_list.append(
                self._extended_source_magnitude_for_each_image(
                    band, source, lensed=lensed
                )
            )
        return magnitude_list

    def extended_source_magnitude(self, band, lensed=False):
        """Unlensed apparent magnitude of the extended source for a given band
        (assumes that size is the same for different bands). This function
        gives gives magnitude for all the provided sources.

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band or list of magnitude
            of each source.
        """
        # band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        magnitude_list = []
        # loop through each source.
        for index, source in enumerate(self._source):
            magnitude_list.append(
                self._extended_source_magnitude(band, source, index, lensed=lensed)
            )
        return magnitude_list

    def _extended_source_magnitude_for_each_image(self, band, source, lensed=False):
        """Extended source magnitude, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function does operation
        only for the single source.

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
            of each image.
        :type lensed: bool
        :return: extended source magnitude of a single source.
        """
        if lensed:
            magnif = self._point_source_magnification(source, extended=True)
            magnif_log = 2.5 * np.log10(abs(magnif))
            source_mag_unlensed = source.extended_source_magnitude(band)
            magnified_mag_list = []
            for i in range(len(magnif_log)):
                magnified_mag_list.append(source_mag_unlensed - magnif_log[i])
            return np.array(magnified_mag_list)
        return source.extended_source_magnitude(band)

    def _extended_source_magnitude(self, band, source, source_index, lensed=False):
        """Unlensed apparent magnitude of the extended source for a given band
        (assumes that size is the same for different bands). This function
        gives magnitude of a single source. Additionally, this function uses
        total magnification to provide a lensed source magnitude.

        :param band: imaging band
        :type band: string
        :param source: Source class instance
        :param source_index: index of a source in source list.
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        # band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        source_mag = source.extended_source_magnitude(band)
        if lensed:
            mag = self._extended_single_source_magnification(source, source_index)
            return source_mag - 2.5 * np.log10(mag)
        return source_mag

    def point_source_magnification(self):
        """Macro-model magnification of point sources. This function calculates
        magnification for each sources.

        :return: list of signed magnification of point sources in same
            order as image positions.
        """
        if not hasattr(self, "_ps_magnification_list"):
            self._ps_magnification_list = []
            for source in self._source:
                self._ps_magnification_list.append(
                    self._point_source_magnification(source)
                )
        return self._ps_magnification_list

    def _point_source_magnification(self, source, extended=False):
        """Macro-model magnification of a point source. This is for a single
        source. The function also works for extended source. For this, It uses
        center of the extended source to calculate lensing magnification.

        :param source: Source class instance. The redshift of this
            source is used in the LensModel.
        :param extended: Boolean. If True, computes the magnification
            for extended source and ignores point source case.
        :return: signed magnification of a point source (extended
            source) in same order as image positions
        """
        lens_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        lensModel = LensModel(
            lens_model_list=lens_model_list,
            z_lens=self.deflector_redshift,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
            z_source=source.redshift,
        )
        if extended is True:
            img_x, img_y = self._extended_source_image_positions(source)
        else:
            img_x, img_y = self._point_source_image_positions(source)
        self._ps_magnification = lensModel.magnification(img_x, img_y, kwargs_lens)
        return self._ps_magnification

    def extended_source_magnification(self):
        """Compute the extended lensed surface brightness and calculates the
        integrated flux-weighted magnification factor of each extended host
        galaxy .

        :return: list of integrated magnification factor of host
            magnitude for each source
        """
        # TODO: add source redshift in ray_shooting. Wait for lenstronomy new version.

        if not hasattr(self, "_extended_source_magnification_list"):
            self._extended_source_magnification_list = []
            for index, source in enumerate(self._source):
                self._extended_source_magnification_list.append(
                    self._extended_single_source_magnification(source, index)
                )
        return self._extended_source_magnification_list

    def extended_source_magnification_for_individual_image(self):
        """Macro-model magnification of extended sources. This function
        calculates magnification for each extended sources at each image
        position.

        :return: list of signed magnification of point sources in same
            order as image positions.
        """
        if not hasattr(self, "_es_magnification_for_each_image_list"):
            self._es_magnification_for_each_image_list = []
            for source in self._source:
                self._es_magnification_for_each_image_list.append(
                    self._point_source_magnification(source, extended=True)
                )
        return self._es_magnification_for_each_image_list

    def _extended_single_source_magnification(self, source, source_index):
        """Compute the extended lensed surface brightness and calculates the
        integrated flux-weighted magnification factor of the extended host
        galaxy. This function does the operation for single source.

        :param source: Source class instance
        :param source_index: index of a source in source list.
        :return: integrated magnification factor of host magnitude
        """
        lens_mass_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
        light_model_list = source.extended_source_light_model()
        kwargs_source_mag = source.kwargs_extended_source_light()
        kwargs_source_mag[0]["center_x"] = self._extended_source_position(source)[0]
        kwargs_source_mag[0]["center_y"] = self._extended_source_position(source)[1]

        lightModel = LightModel(light_model_list=light_model_list)
        lensModel = LensModel(
            lens_model_list=lens_mass_model_list,
            z_lens=self.deflector_redshift,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
            z_source=source.redshift,
        )
        theta_E = self._einstein_radius(source)
        center_source = self._extended_source_position(source)

        kwargs_source_amp = data_util.magnitude2amplitude(
            lightModel, kwargs_source_mag, magnitude_zero_point=0
        )

        num_pix = 200
        delta_pix = theta_E * 4 / num_pix
        x, y = util.make_grid(numPix=num_pix, deltapix=delta_pix)
        x += center_source[0]
        y += center_source[1]
        beta_x, beta_y = lensModel.ray_shooting(x, y, kwargs_lens)
        flux_lensed = np.sum(
            lightModel.surface_brightness(beta_x, beta_y, kwargs_source_amp)
        )
        flux_no_lens = np.sum(lightModel.surface_brightness(x, y, kwargs_source_amp))
        if flux_no_lens > 0:
            self._extended_source_magnification = flux_lensed / flux_no_lens
        else:
            self._extended_source_magnification = 0
        return self._extended_source_magnification

    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-
            normalized amplitudes
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
        }
        if self.source_number > 1:
            kwargs_model["lens_redshift_list"] = [self.deflector_redshift] * len(
                lens_mass_model_list
            )
            kwargs_model["z_lens"] = self.deflector_redshift
            if self.max_redshift_source_class.light_profile in [
                "single_sersic",
                "interpolated",
            ]:
                kwargs_model["source_redshift_list"] = self.source_redshift_list
            elif self.max_redshift_source_class.light_profile == "double_sersic":
                kwargs_model["source_redshift_list"] = [
                    z for z in self.source_redshift_list for _ in range(2)
                ]
            kwargs_model["z_source_convention"] = (
                self.max_redshift_source_class.redshift
            )
            kwargs_model["z_source"] = self.max_redshift_source_class.redshift
            kwargs_model["cosmo"] = self.cosmo

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
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :return: lens_model_list, kwargs_lens
        """
        if hasattr(self, "_lens_mass_model_list") and hasattr(self, "_kwargs_lens"):
            return self._lens_mass_model_list, self._kwargs_lens
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
        kappa_ext, gamma1, gamma2 = self.los_linear_distortions
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
        self._kwargs_lens = kwargs_lens
        self._lens_mass_model_list = lens_mass_model_list

        return lens_mass_model_list, kwargs_lens

    def deflector_light_model_lenstronomy(self, band):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

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
            source_models_list = []
            kwargs_source_list = []
            for source in self._source:
                source_models_list.append(source.extended_source_light_model())
                kwargs_es_light = source.kwargs_extended_source_light(band=band)
                kwargs_es_light[0]["center_x"] = self._extended_source_position(source)[0]
                kwargs_es_light[0]["center_y"] = self._extended_source_position(source)[1]
                kwargs_source_list.append(kwargs_es_light)
            # lets transform list in to required structure
            """if (
                self.max_redshift_source_class.light_profile == "double_sersic"
                and self.source_number > 1
            ):
                source_models_list_restructure = source_models_list
                kwargs_source_list_restructure = kwargs_source_list
            else:"""
            source_models_list_restructure = list(np.concatenate(source_models_list))
            kwargs_source_list_restructure = list(np.concatenate(kwargs_source_list))
            source_models["source_light_model_list"] = source_models_list_restructure
            kwargs_source = kwargs_source_list_restructure
        else:
            # source_models['source_light_model_list'] = None
            kwargs_source = None

        if (
            self._source_type == "point_source"
            or self._source_type == "point_plus_extended"
        ):
            source_models_list = []
            kwargs_ps_list = []
            for source in self._source:
                source_models_list.append("LENSED_POSITION")
                img_x, img_y = self._point_source_image_positions(source=source)
                if band is None:
                    image_magnitudes = np.abs(self._point_source_magnification(source))
                else:
                    image_magnitudes = self._point_source_magnitude(
                        band=band, source=source, lensed=True
                    )
                kwargs_ps_list.append(
                    {
                        "ra_image": img_x,
                        "dec_image": img_y,
                        "magnitude": image_magnitudes,
                    }
                )
            source_models["point_source_model_list"] = source_models_list
            kwargs_ps = kwargs_ps_list
        else:
            # source_models['point_source_model'] = None
            kwargs_ps = None
        all_source_kwarg_dict["kwargs_source"] = kwargs_source
        all_source_kwarg_dict["kwargs_ps"] = kwargs_ps
        return source_models, all_source_kwarg_dict

    def kappa_star(self, ra, dec):
        """Computes the stellar surface density at location (ra, dec) in units
        of lensing convergence.

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

    def contrast_ratio(self, band, source_index=0):
        """Computes the surface brightness ratio (difference in magnitude per
        arc second square) at image positions of the source, for the source as
        the average surface brightness within the half light radius, for the
        lens light at the position of the lensed images.

        :param source_index: index of source, default =0, i.e. the first
            source
        :type source_index: int
        :param band: bandpass filter
        :type: str
        :return: surface brightness ratio for all images
            I_source_light/I_lens_light [mag/arcsec^2]
        """
        # TODO: make a definition that is more flexible to use either point sources or extended sources
        ra, dec = self.extended_source_image_positions()[source_index]

        mag_arcsec2_lens_light = self.deflector.surface_brightness(ra, dec, band=band)
        mag_arcsec2_source = self.source(source_index).surface_brightness_reff(
            band=band
        )

        return mag_arcsec2_source - mag_arcsec2_lens_light

    def generate_id(self, ra=None, dec=None):
        """Generate a unique ID for the lens based on its position.

        :param ra: ra coordinate of the Lens
        :param dec: dec coordinate of the Lens
        :return: A string representing the lens ID.
        """
        if ra is None and dec is None:
            ra = self.deflector_position[0]
            dec = self.deflector_position[1]
        else:
            ra = ra
            dec = dec
        if self._source_type == "extended":
            lens_type = "GG"
        elif (
            self._source_type == "point_source"
            or self._source_type == "point_plus_extended"
        ):
            if self.max_redshift_source_class.sn_type is not None:
                lens_type = "SN" + self.max_redshift_source_class.sn_type
            else:
                lens_type = "QSO"

        return f"{lens_type}-LENS_{ra:.4f}_{dec:.4f}"


def image_separation_from_positions(image_positions):
    """Calculate image separation in arc-seconds; if there are only two images,
    the separation between them is returned; if there are more than 2 images,
    the maximum separation is returned.

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
