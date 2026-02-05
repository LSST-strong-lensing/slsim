import numpy as np
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.lens_equation_solver import (
    analytical_lens_model_support,
)

from slsim.Util.param_util import (
    ellipticity_slsim_to_lenstronomy,
    image_separation_from_positions,
)
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import data_util
from lenstronomy.Util import util
from slsim.Util.catalog_util import safe_value

from slsim.Lenses.lensed_system_base import LensedSystemBase
from slsim.Deflectors.deflector import JAX_PROFILES
import pandas as pd
from copy import deepcopy


class Lens(LensedSystemBase):
    """Class to manage individual lenses."""

    def __init__(
        self,
        source_class,
        deflector_class,
        cosmo,
        lens_equation_solver="lenstronomy_analytical",
        magnification_limit=0.01,
        los_class=None,
        use_jax=True,
        multi_plane=None,
        shear=True,
        convergence=True,
    ):
        """

        :param source_class: A Source class instance or list of Source class instance
        :type source_class: Source class instance from slsim.Sources.source.
         See the Source class documentation.
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
         See the Deflector class documentation.
        :param cosmo: astropy.cosmology instance
        :param lens_equation_solver: type of lens equation solver; currently supporting
         "lenstronomy_analytical" and "lenstronomy_general"
        :type lens_equation_solver: str
        :param magnification_limit: absolute lensing magnification lower limit to
            register a point source (ignore highly de-magnified images)
        :type magnification_limit: float >= 0
        :param los_class: line of sight dictionary (optional, takes these values instead of drawing from distribution)
        :type los_class: ~LOSIndividual() class object
        :param use_jax: if True, will use JAX version of lenstronomy to do lensing calculations for models that are
            supported in JAXtronomy
        :type use_jax: bool
        :param multi_plane: None for single-plane, 'Source' for multi-source plane, 'Deflector' for multi-deflector plane,
            or 'Both' for both multi-deflector and multi-source plane
        :type multi_plane: None or str
        :param shear: whether to include external shear in multi-plane lensing
        :type shear: bool
        :param convergence: whether to include external convergence in multi-plane lensing
        :type convergence: bool

        """
        LensedSystemBase.__init__(
            self,
            source_class=source_class,
            deflector_class=deflector_class,
            los_class=los_class,
            multi_plane=multi_plane,
            shear=shear,
            convergence=convergence,
        )
        # SourceList.__init__(self, source_class_list=source_class)
        self.cosmo = cosmo
        self._lens_equation_solver = lens_equation_solver
        self._magnification_limit = magnification_limit

        # we conventionally use the highest source redshift in the lens cosmo.
        self._lens_cosmo = LensCosmo(
            z_lens=self.deflector.redshift,
            z_source=self.max_redshift_source_class.redshift,
            cosmo=self.cosmo,
        )
        self._use_jax = use_jax

    def source(self, index=0):
        """

        :param index: index of the source
        :type index: int
        :return: Source() class with index
        """
        return self._source[index]

    @property
    def image_number(self):
        """Number of images in the lensing configuration. If a point source is
        present, uses point source, otherwise extended source.

        :return: number of images for each source model
        """
        n_image = [len(pos[0]) for pos in self.point_source_image_positions()]
        return n_image

    @property
    def deflector_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        return self.deflector.deflector_center

    @property
    def extended_source_image_positions(self):
        """Returns extended source image positions by solving the lens equation
        for each source.

        :return: list of (x-pos, y-pos)
        """
        if not hasattr(self, "_es_image_position_list"):
            self._es_image_position_list = []
            for index in range(len(self._source)):
                self._es_image_position_list.append(
                    self._extended_source_image_positions(index)
                )
        return self._es_image_position_list

    def _extended_source_image_positions(self, source_index):
        """Returns extended source image positions by solving the lens equation
        for a single source.

        :param source_index: index of a source in source list.
        :return: x-pos, y-pos
        """
        source_pos_x, source_pos_y = self.source(source_index).extended_source_position
        return self._image_position_from_source(
            source_pos_x, source_pos_y, source_index
        )

    def point_source_image_positions(self):
        """Returns point source image positions by solving the lens equation
        for all sources. In the absence of a point source, this function
        returns the solution for the center of the extended source.

        :return: list of (x-pos, y-pos) for each source
        """
        if not hasattr(self, "_ps_image_position_list"):
            self._ps_image_position_list = []
            for index in range(len(self._source)):
                self._ps_image_position_list.append(
                    self._point_source_image_positions(index)
                )
        return self._ps_image_position_list

    def _point_source_image_positions(self, source_index):
        """Returns point source image positions by solving the lens equation
        for a single source. In the absence of a point source, this function
        returns the solution for the center of the extended source.

        :param source_index: index of a source in source list.
        :return: x-pos, y-pos
        """
        point_source_pos_x, point_source_pos_y = self.source(
            source_index
        ).point_source_position
        return self._image_position_from_source(
            point_source_pos_x, point_source_pos_y, source_index
        )

    def _image_position_from_source(self, x_source, y_source, source_index):
        """Solves the lens equation.

        :param x_source: x-position of source [arc-seconds]
        :param y_source: y-position of source [arc-seconds]
        :param source_index: source index
        :return: solution of the lens equation for source position [RA
            list], [DEC list]
        """
        lens_model_class, kwargs_lens = self.deflector_mass_model_lenstronomy(
            source_index=source_index
        )
        lens_eq_solver = LensEquationSolver(lens_model_class)
        point_source_pos_x, point_source_pos_y = x_source, y_source
        # uses analytical lens equation solver in case it is supported by lenstronomy for speed-up
        if (
            self._lens_equation_solver == "lenstronomy_analytical"
            and analytical_lens_model_support(lens_model_class.lens_model_list) is True
        ):
            solver = "analytical"
        else:
            solver = "lenstronomy"
        einstein_radius = self._approximate_einstein_radius(source_index=source_index)
        image_positions = lens_eq_solver.image_position_from_source(
            point_source_pos_x,
            point_source_pos_y,
            kwargs_lens,
            solver=solver,
            search_window=einstein_radius * 6,
            min_distance=einstein_radius * 6 / 100,
            magnification_limit=self._magnification_limit,
            num_iter_max=25,
        )

        return image_positions

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
            magnitude of the second-brightest image and corresponding
            band. If provided, selects lenses where the second-brightest
            image has a magnitude less than or equal to provided
            magnitude. e.g.: second_brightest_image_cut = {"i": 23, "g":
            24, "r": 22}
        :return: A boolean or dict of boolean.
        """
        validity_results = {}
        for index in range(len(self._source)):
            validity_results[index] = self._validity_test(
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
        min_image_separation=0,
        max_image_separation=10,
        mag_arc_limit=None,
        second_brightest_image_cut=None,
        source_index=0,
    ):
        """Check whether a single lensing configuration matches selection and
        plausibility criteria.

        :param min_image_separation: minimum image separation
        :param max_image_separation: maximum image separation
        :param mag_arc_limit: dictionary with key of bands and values of
            magnitude limits of integrated lensed arc
        :type mag_arc_limit: dict with key of bands and values of
            magnitude limits
        :param second_brightest_image_cut: Dictionary containing maximum
            magnitude of the second-brightest image and corresponding
            band. If provided, selects lenses where the second-brightest
            image has a magnitude less than or equal to provided
            magnitude.
            eg: second_brightest_image_cut = {"i": 23, "g": 24, "r": 22}
        :param source_index: index of a source in source list.
        :return: boolean
        """

        # Criteria 1:The redshift of the lens (z_lens) must be less than the
        # redshift of the source (z_source).
        z_lens = np.max(self.deflector_redshift)
        z_source = self.source(source_index).redshift
        if z_lens >= z_source:
            return False

        # Criteria 2: The angular Einstein radius of the lensing configuration (theta_E)
        # times 2 must be greater than or equal to the minimum image separation
        # (min_image_separation) and less than or equal to the maximum image
        # separation (max_image_separation).
        einstein_radius = self._approximate_einstein_radius(source_index=source_index)
        if not min_image_separation <= 2 * einstein_radius <= max_image_separation:
            return False

        # Criteria 3: The distance between the lens center and the source position
        # must be less than or equal to the angular Einstein radius
        # of the lensing configuration (times sqrt(2)).
        source_pos = self.source(source_index).point_source_position
        center_lens, center_source = (self.deflector_position, source_pos)
        if np.sum((center_lens - center_source) ** 2) > einstein_radius**2 * 2:
            return False

        # Criteria 4: The lensing configuration must produce at least two SL images.
        image_positions = self.point_source_image_positions()[source_index]
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
        if mag_arc_limit is not None:
            # makes sure magnification of extended source is only used when there is
            # an extended source
            bool_mag_limit = False
            host_mag = self._extended_integrated_source_magnification(source_index)
            if host_mag is not None:
                for band, mag_limit_band in mag_arc_limit.items():
                    mag_source = self._extended_source_magnitude(band, source_index)
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
                if self.source(source_index).source_type == "extended":
                    image_magnitude_list = (
                        self.extended_source_magnitude_for_each_image(
                            band=band_max, lensed=True
                        )
                    )
                else:
                    image_magnitude_list = self.point_source_magnitude(
                        band=band_max, lensed=True
                    )
                second_brightest_mag = np.sort(image_magnitude_list[source_index])[1]
                if second_brightest_mag > mag_max:
                    return False
        return True
        # TODO: test for signal-to-noise ratio in surface brightness

    @property
    def deflector_redshift(self):
        """

        :return: lens redshift

        """
        deflector_redshifts = [self.deflector.redshift]

        if self.multi_plane or self.source_number > 1:

            if self.deflector.deflector_type in ["NFW_CLUSTER"]:

                if self.deflector.cored_profile:
                    deflector_redshifts.append(self.deflector.redshift)

                deflector_redshifts.extend(self.deflector.subhalo_redshifts)

            if self.shear:
                deflector_redshifts.append(self.deflector.redshift)

            if self.convergence:
                deflector_redshifts.append(self.deflector.redshift)

            return deflector_redshifts
        else:
            return deflector_redshifts[0]

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
            for index in range(len(self._source)):
                self._theta_E_list.append(self._einstein_radius(index))
        return self._theta_E_list

    @property
    def einstein_radius_infinity(self):
        """Einstein radius when source is at infinity.

        :return: Einstein radius of a deflector.
        """

        if not hasattr(self, "_theta_E_infinity"):
            self._theta_E_infinity = self.deflector.theta_e_infinity(
                self.cosmo, multi_plane=self.multi_plane, use_jax=self._use_jax
            )
        return self._theta_E_infinity

    def _approximate_einstein_radius(self, source_index):
        """Returns the appropriate Einstein radius depending on the deflector
        type. This definition is meant to estimate an approximate
        (conservative) area to place a source.

        :param source_index: index of the source.
        :return: effective Einstein radius for the lens-source pair.
        """
        if self.deflector.deflector_type in ["EPL", "EPL_SERSIC"]:
            return self.einstein_radius[source_index]
        else:
            return self.einstein_radius_infinity

    def _einstein_radius(self, source_index):
        """Einstein radius, including external shear.

        :param source_index: index of a source in source list.
        :return: einstein radius of a lens-source pair.
        """
        if self.deflector.redshift >= self.source(source_index).redshift:
            theta_E = 0
            return theta_E
        lens_model_class, kwargs_lens = self.deflector_mass_model_lenstronomy(
            source_index=source_index
        )
        if self.deflector.deflector_type in ["EPL", "EPL_SERSIC"]:
            kappa_ext_convention = self.los_class.convergence
            gamma_pl = self.deflector.halo_properties["gamma_pl"]
            theta_E_convention = kwargs_lens[0]["theta_E"]
            if (
                self.source(source_index).redshift
                == self.max_redshift_source_class.redshift
            ):
                theta_E = theta_E_convention
                kappa_ext = kappa_ext_convention
            else:
                beta = self._lens_cosmo.beta_double_source_plane(
                    z_lens=self.deflector.redshift,
                    z_source_2=self.max_redshift_source_class.redshift,
                    z_source_1=self.source(source_index).redshift,
                )

                theta_E = theta_E_convention * beta ** (1.0 / (gamma_pl - 1))
                kappa_ext = kappa_ext_convention * beta

            theta_E /= (1 - kappa_ext) ** (1.0 / (gamma_pl - 1))

        else:
            # numerical solution for the Einstein radius
            lens_analysis = LensProfileAnalysis(lens_model=lens_model_class)
            theta_E = lens_analysis.effective_einstein_radius(
                kwargs_lens, r_min=1e-3, r_max=5e1, num_points=100
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
        for index in range(len(self._source)):
            arrival_times_list.append(self._point_source_arrival_times(index))
        return arrival_times_list

    def _point_source_arrival_times(self, source_index):
        """Arrival time of images relative to a straight line without lensing.
        Negative values correspond to images arriving earlier, and positive
        signs correspond to images arriving later.

        :param source_index: index of a source in source list.
        :return: arrival times for each image [days]
        :rtype: numpy array
        """
        lens_model, kwargs_lens = self.deflector_mass_model_lenstronomy(
            source_index=source_index
        )

        x_image, y_image = self._point_source_image_positions(source_index=source_index)
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
        for index in range(len(self._source)):
            observer_times_list.append(self._image_observer_times(index, t_obs))
        if self.source_number == 1:
            return observer_times_list[0]
        return observer_times_list

    def _image_observer_times(self, source_index, t_obs):
        """Calculates time of a source at the different images, not correcting
        for redshifts, but for time delays. The time is relative to the first
        arriving image.

        :param source_index: index of a source in source list.
        :param t_obs: time of observation [days]. It could be a single
            observation time or an array of observation time.
        :return: time of the source when seen in the different images
            (without redshift correction)
        :rtype: numpy array. Each element of the array corresponds to
            different image observation times.
        """
        arrival_times = self._point_source_arrival_times(source_index)
        if type(t_obs) is np.ndarray and len(t_obs) > 1:
            observer_times = (
                t_obs[:, np.newaxis] - arrival_times + np.min(arrival_times)
            ).T
        else:
            observer_times = (t_obs - arrival_times + np.min(arrival_times))[
                :, np.newaxis
            ]

        return observer_times

    def point_source_magnitude(
        self,
        band,
        lensed=False,
        time=None,
        microlensing=False,
        kwargs_microlensing=None,
    ):
        """Point source magnitude, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function provided
        magnitudes of all the sources.

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :param time: time is an image observation time in units of days.
            If None, provides magnitude without variability.
        :param microlensing: if using micro-lensing map to produce the
            lensed magnification
        :type microlensing: bool
        :param kwargs_microlensing: additional (optional) dictionary of
            settings required by micro-lensing calculation that do not
            depend on the Lens() class. It is of type:
            kwargs_microlensing = {"kwargs_magnification_map":
            kwargs_magnification_map, "point_source_morphology":
            'gaussian' or 'agn' or 'supernovae',
            "kwargs_source_morphology": kwargs_source_morphology} The
            kwargs_source_morphology is required for the source
            morphology calculation. The kwargs_magnification_map is
            required for the microlensing calculation. See the classes
            in slsim.Microlensing for more details on the
            kwargs_magnification_map and kwargs_source_morphology.
        :type kwargs_microlensing: dict
        :return: list of point source magnitudes.
        """

        magnitude_list = []
        for index in range(len(self._source)):
            magnitude_list.append(
                self._point_source_magnitude(
                    band,
                    source_index=index,
                    lensed=lensed,
                    time=time,
                    microlensing=microlensing,
                    kwargs_microlensing=kwargs_microlensing,
                )
            )
        return magnitude_list

    def _point_source_magnitude(
        self,
        band,
        source_index,
        lensed=False,
        time=None,
        microlensing=False,
        kwargs_microlensing=None,
    ):
        """Point source magnitude, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function does operation
        only for the single source.

        :param band: imaging band
        :type band: string
        :param source_index: index of a source in source list.
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :param time: time is a image observation time in units of days.
            If None, provides magnitude without variability.
        :param microlensing: to include microlensing effect?
        :type microlensing: bool
        :param kwargs_microlensing: additional (optional) dictionary of
            settings required by micro-lensing calculation that do not
            depend on the Lens() class. It is of type:
            kwargs_microlensing = {"kwargs_magnification_map":
            kwargs_magnification_map, "point_source_morphology":
            'gaussian' or 'agn' or 'supernovae',
            "kwargs_source_morphology": kwargs_source_morphology} The
            kwargs_source_morphology is required for the source
            morphology calculation. The kwargs_magnification_map is
            required for the microlensing calculation. See the classes
            in slsim.Microlensing for more details on the
            kwargs_magnification_map and kwargs_source_morphology.
        :type kwargs_microlensing: dict
        :return: point source magnitude of a single source
        """
        # TODO: might have to change conventions between extended and point source
        if lensed:
            magnif = self._point_source_magnification(source_index)
            magnif_log = 2.5 * np.log10(abs(magnif))
            if time is not None:
                time = time
                image_observed_times = self._image_observer_times(source_index, time)
                variable_magnitude = self.source(source_index).point_source_magnitude(
                    band,
                    image_observation_times=image_observed_times,
                )
                lensed_variable_magnitude = (
                    variable_magnitude - magnif_log[:, np.newaxis]
                )
                if microlensing:
                    microlensing_magnitudes = self._point_source_magnitude_microlensing(
                        band=band,
                        time=time,
                        source_index=source_index,
                        kwargs_microlensing=kwargs_microlensing,
                    )
                    lensed_variable_magnitude += microlensing_magnitudes

                return lensed_variable_magnitude

            else:
                source_mag_unlensed = self.source(source_index).point_source_magnitude(
                    band
                )
                magnified_mag_list = []
                for i in range(len(magnif_log)):
                    magnified_mag_list.append(source_mag_unlensed - magnif_log[i])
                return np.array(magnified_mag_list)
        return self.source(source_index).point_source_magnitude(band)

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
            of each image.
        :type lensed: bool
        :return: list of extended source magnitudes.
        """

        magnitude_list = []
        for index in range(len(self._source)):
            magnitude_list.append(
                self._extended_source_magnitude_for_each_image(
                    band, source_index=index, lensed=lensed
                )
            )
        return magnitude_list

    def _microlensing_parameters_for_image_positions_single_source(
        self, band, source_index
    ):
        """For a given source, calculates the microlensing parameters for each
        image position.

        :param band: imaging band
        :type band: string
        :param source_index: index of a source in source list.
        :return: kappa_star, kappa_tot, shear, shear_angle kappa_star is
            the stellar convergence, kappa_tot is the total convergence,
            shear is the magnitude of the shear vector, and shear_angle
            is the angle of shear vector in radians. The returned arrays
            contains the values for each image of the source in the
            lensing configuration. The arrays are of the same length as
            the number of images of the source.
        :rtype: tuple of numpy arrays
        """
        lenstronomy_kwargs = self.lenstronomy_kwargs(band=band)
        lens_model_lenstronomy = LensModel(
            lens_model_list=lenstronomy_kwargs[0]["lens_model_list"]
        )
        lenstronomy_kwargs_lens = lenstronomy_kwargs[1]["kwargs_lens"]

        image_positions_x, image_positions_y = self._point_source_image_positions(
            source_index
        )

        kappa_star_images = self.kappa_star(image_positions_x, image_positions_y)
        kappa_tot_images = lens_model_lenstronomy.kappa(
            image_positions_x, image_positions_y, lenstronomy_kwargs_lens
        )
        gamma1, gamma2 = lens_model_lenstronomy.gamma(
            image_positions_x, image_positions_y, lenstronomy_kwargs_lens
        )

        shear_images = np.sqrt(gamma1**2 + gamma2**2)
        shear_angle_images = np.arctan2(gamma2, gamma1)

        return kappa_star_images, kappa_tot_images, shear_images, shear_angle_images

    def _point_source_magnitude_microlensing(
        self, band, time, source_index, kwargs_microlensing=None
    ):
        """Returns point source magnitude variability from only microlensing
        effect. This function does operation only for the single source.

        :param band: imaging band
        :type band: string
        :param time: time is an image observation time in units of days.
        :param kwargs_microlensing (Optional): additional dictionary of
            settings required by micro-lensing calculation. It is of
            type: kwargs_microlensing = {"kwargs_magnification_map":
            kwargs_magnification_map, "point_source_morphology":
            'gaussian' or 'agn' or 'supernovae',
            "kwargs_source_morphology": kwargs_source_morphology} The
            kwargs_source_morphology is required for the source
            morphology calculation. The kwargs_magnification_map is
            required for the microlensing calculation. If None, defaults
            are used corresponding to the source in the lens class.
        :type kwargs_microlensing: dict or None
        :return: point source magnitude for a single source, does not
            include the macro-magnification.
        :rtype: numpy array
        """
        # importing here to keep it optional
        from slsim.Microlensing.lightcurvelensmodel import (
            MicrolensingLightCurveFromLensModel,
        )

        # get microlensing parameters
        kappa_star_images, kappa_tot_images, shear_images, shear_angle_images_rad = (
            self._microlensing_parameters_for_image_positions_single_source(
                band=band, source_index=source_index
            )
        )
        # convert shear angle to degrees for the microlensing class
        shear_phi_angle_images = np.degrees(shear_angle_images_rad)

        # select random RA and DEC in Sky for the lens,
        # #TODO: In future, this should be the position of the lens in the sky
        ra_lens = np.random.uniform(0, 360)  # degrees
        dec_lens = np.random.uniform(-90, 90)  # degrees

        ##########################################################################
        ## Update kwargs_microlensing from source class
        ##########################################################################
        if kwargs_microlensing is None:
            kwargs_microlensing_updated = {}
        else:
            # Make a copy of kwargs_microlensing to avoid modifying the original dict
            kwargs_microlensing_updated = deepcopy(kwargs_microlensing)

        # Get or initialize kwargs_source_morphology
        if "kwargs_source_morphology" not in kwargs_microlensing_updated:
            kwargs_source_morphology = {}
        else:
            kwargs_source_morphology = kwargs_microlensing_updated[
                "kwargs_source_morphology"
            ]

        # Update kwargs_source_morphology with values from the Lens class if not provided by the user
        if "source_redshift" not in kwargs_source_morphology:
            kwargs_source_morphology["source_redshift"] = self.source(
                source_index
            ).redshift
        if "cosmo" not in kwargs_source_morphology:
            kwargs_source_morphology["cosmo"] = self.cosmo
        if "observing_wavelength_band" not in kwargs_source_morphology:
            kwargs_source_morphology["observing_wavelength_band"] = band

        # Extract additional parameters from the source class if not provided
        kwargs_source_morphology = self.source(
            source_index
        )._source.update_microlensing_kwargs_source_morphology(kwargs_source_morphology)

        # Update the main microlensing kwargs dictionary
        kwargs_microlensing_updated["kwargs_source_morphology"] = (
            kwargs_source_morphology
        )

        # Update point_source_morphology based on source type
        if "point_source_morphology" not in kwargs_microlensing_updated:
            if self.source(source_index)._source.name == "QSO":
                kwargs_microlensing_updated["point_source_morphology"] = "agn"
        ##########################################################################

        # Instantiate the microlensing model with all required parameters
        # Check if the microlensing model class is already instantiated for this source index to avoid redundant instantiation
        if not hasattr(self, "_microlensing_model_class"):
            self._microlensing_model_class = {}

        if source_index not in self._microlensing_model_class.keys():
            self._microlensing_model_class[source_index] = (
                MicrolensingLightCurveFromLensModel(
                    source_redshift=self.source(source_index).redshift,
                    deflector_redshift=self.deflector_redshift,
                    kappa_star_images=kappa_star_images,
                    kappa_tot_images=kappa_tot_images,
                    shear_images=shear_images,
                    shear_phi_angle_images=shear_phi_angle_images,
                    ra_lens=ra_lens,
                    dec_lens=dec_lens,
                    deflector_velocity_dispersion=self.deflector_velocity_dispersion(),
                    cosmology=self.cosmo,
                    **kwargs_microlensing_updated,
                )
            )
        else:
            # Update existing instance with new parameters if needed
            self._microlensing_model_class[source_index].update_source_morphology(
                kwargs_source_morphology
            )

        # Generate microlensing magnitudes with the simplified method call
        microlensing_magnitudes = self._microlensing_model_class[
            source_index
        ].generate_point_source_microlensing_magnitudes(time=time)

        return microlensing_magnitudes  # # does not include the macro-lensing effect

    def microlensing_model_class(self, source_index):
        """Returns the MicrolensingLightCurveFromLensModel class instance
        corresponding to a specific source index for the microlensing
        calculations. Only available if microlensing=True was used in
        point_source_magnitude.

        :param source_index: index of a source in source list.
        :return: MicrolensingLightCurveFromLensModel class instance for
            the specified source.
        """
        if hasattr(self, "_microlensing_model_class"):
            if source_index not in self._microlensing_model_class:
                raise AttributeError(
                    f"MicrolensingLightCurveFromLensModel class is not set for source index {source_index}. "
                    "Please run point_source_magnitude with microlensing=True."
                )
            return self._microlensing_model_class[source_index]
        else:
            raise AttributeError(
                "MicrolensingLightCurveFromLensModel class is not set. "
                "Please run point_source_magnitude with microlensing=True."
            )

    def extended_source_magnitude(self, band, lensed=False):
        """Unlensed apparent magnitude of the extended source for a given band
        (assumes that size is the same for different bands). This function
        gives magnitude for all the provided sources.

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
        for index in range(len(self._source)):
            magnitude_list.append(
                self._extended_source_magnitude(band, index, lensed=lensed)
            )
        return magnitude_list

    def _extended_source_magnitude_for_each_image(
        self, band, source_index, lensed=False
    ):
        """Extended source magnitude, either unlensed (single value) or lensed
        (array) with macro-model magnifications. This function does operation
        only for the single source.

        :param band: imaging band
        :type band: string
        :param source_index: index of a source in source list.
        :param lensed: if True, returns the lensed magnified magnitude
            of each image.
        :type lensed: bool
        :return: extended source magnitude of a single source.
        """
        if lensed:
            magnif = self._point_source_magnification(
                source_index=source_index, extended=True
            )
            magnif_log = 2.5 * np.log10(abs(magnif))
            source_mag_unlensed = self.source(source_index).extended_source_magnitude(
                band
            )
            magnified_mag_list = []
            for i in range(len(magnif_log)):
                magnified_mag_list.append(source_mag_unlensed - magnif_log[i])
            return np.array(magnified_mag_list)
        return self.source(source_index).extended_source_magnitude(band)

    def _extended_source_magnitude(self, band, source_index, lensed=False):
        """Unlensed apparent magnitude of the extended source for a given band
        (assumes that size is the same for different bands). This function
        gives magnitude of a single source. Additionally, this function uses
        total magnification to provide a lensed source magnitude.

        :param band: imaging band
        :type band: string
        :param source_index: index of a source in source list.
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        # band_string = str("mag_" + band)
        # TODO: might have to change conventions between extended and point source
        source_mag = self.source(source_index).extended_source_magnitude(band)
        if lensed and source_mag is not None:
            mag = self._extended_integrated_source_magnification(
                source_index=source_index
            )
            return source_mag - 2.5 * np.log10(mag)
        return source_mag

    def point_source_magnification(self):
        """Macro-model magnification of point sources. This function calculates
        magnification for each source.

        :return: list of signed magnification of point sources in same
            order as image positions.
        """
        if not hasattr(self, "_ps_magnification_list"):
            self._ps_magnification_list = []
            for index in range(len(self._source)):
                self._ps_magnification_list.append(
                    self._point_source_magnification(source_index=index)
                )
        return self._ps_magnification_list

    def _point_source_magnification(self, source_index, extended=False):
        """Macro-model magnification of a point source. This is for a single
        source. The function also works for extended source. For this, It uses
        center of the extended source to calculate lensing magnification.

        :param source_index: index of a source in source list.
        :param extended: if True, calculates image positions of extended
            source
        :return: signed magnification of a point source (extended
            source) in same order as image positions
        """
        lens_model_class, kwargs_lens = self.deflector_mass_model_lenstronomy(
            source_index=source_index
        )

        if extended:
            img_x, img_y = self._extended_source_image_positions(
                source_index=source_index
            )
        else:
            img_x, img_y = self._point_source_image_positions(source_index=source_index)
        ps_magnification = lens_model_class.magnification(img_x, img_y, kwargs_lens)
        return ps_magnification

    @property
    def extended_source_magnification(self):
        """Compute the extended lensed surface brightness and calculates the
        integrated flux-weighted magnification factor of each extended host
        galaxy .

        :return: list of integrated magnification factor of host
            magnitude for each source
        """

        if not hasattr(self, "_extended_source_magnification_list"):
            self._extended_source_magnification_list = []
            for index in range(len(self._source)):
                self._extended_source_magnification_list.append(
                    self._extended_integrated_source_magnification(source_index=index)
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
            for index in range(len(self._source)):
                self._es_magnification_for_each_image_list.append(
                    self._point_source_magnification(source_index=index, extended=True)
                )
        return self._es_magnification_for_each_image_list

    def _extended_integrated_source_magnification(self, source_index):
        """Compute the extended lensed surface brightness and calculates the
        integrated flux-weighted magnification factor of the extended host
        galaxy. This function does the operation for single source.

        :param source_index: index of a source in source list.
        :return: integrated magnification factor of host magnitude
        """
        light_model_list, kwargs_source_mag = self.source(
            source_index
        ).kwargs_extended_light()
        if len(light_model_list) == 0:
            return None  # no extended source profile
        lens_model_class, kwargs_lens = self.deflector_mass_model_lenstronomy(
            source_index=source_index
        )
        lightModel = LightModel(light_model_list=light_model_list)
        theta_E = self._approximate_einstein_radius(source_index=source_index)
        center_source = self.source(source_index).extended_source_position

        kwargs_source_amp = data_util.magnitude2amplitude(
            lightModel, kwargs_source_mag, magnitude_zero_point=0
        )
        # TODO: this does not work well for clusters when the lensed source is relatively small compared to the image
        num_pix = 200
        delta_pix = theta_E * 4 / num_pix
        x, y = util.make_grid(numPix=num_pix, deltapix=delta_pix)
        x += center_source[0]
        y += center_source[1]
        beta_x, beta_y = lens_model_class.ray_shooting(x, y, kwargs_lens)
        # test conventions
        flux_lensed = np.sum(
            lightModel.surface_brightness(beta_x, beta_y, kwargs_source_amp)
        )
        flux_no_lens = np.sum(lightModel.surface_brightness(x, y, kwargs_source_amp))
        if flux_no_lens > 0:
            extended_source_magnification = flux_lensed / flux_no_lens
        else:
            extended_source_magnification = 0
        return extended_source_magnification

    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-
            normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        lens_model, kwargs_lens = self.deflector_mass_model_lenstronomy(source_index=0)
        lens_model_list = lens_model.lens_model_list
        # TODO: extract other potentially relevant keyword arguments (such as redshift list, multi-plane etc)

        (
            lens_light_model_list,
            kwargs_lens_light,
        ) = self.deflector.light_model_lenstronomy(band=band)
        # list of

        kwargs_model = {
            "lens_light_model_list": lens_light_model_list,
            "lens_model_list": lens_model_list,
        }

        if self.multi_plane or self.source_number > 1:

            kwargs_model["lens_redshift_list"] = self.deflector_redshift
            kwargs_model["z_lens"] = self.deflector.redshift
            kwargs_model["z_source"] = self.max_redshift_source_class.redshift
            kwargs_model["cosmo"] = self.cosmo

            if self.max_redshift_source_class.extended_source_type in [
                "single_sersic",
                "interpolated",
            ]:
                kwargs_model["source_redshift_list"] = self.source_redshift_list
            elif self.max_redshift_source_class.extended_source_type in [
                "double_sersic"
            ]:
                kwargs_model["source_redshift_list"] = [
                    z for z in self.source_redshift_list for _ in range(2)
                ]
            kwargs_model["z_source_convention"] = (
                self.max_redshift_source_class.redshift
            )

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

    def deflector_mass_model_lenstronomy(self, source_index=None):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :return: LensModel() class, kwargs_lens
        """
        if source_index is None:
            z_source = self.max_redshift_source_class.redshift
        else:
            z_source = self.source(source_index).redshift
        if hasattr(self, "_lens_mass_model_list") and hasattr(self, "_kwargs_lens"):
            pass
        elif self.deflector.deflector_type in [
            "EPL",
            "EPL_SERSIC",
            "NFW_HERNQUIST",
            "NFW_CLUSTER",
        ]:

            lens_mass_model_list, kwargs_lens = self.deflector.mass_model_lenstronomy(
                lens_cosmo=self._lens_cosmo
            )
            # adding line-of-sight structure
            kappa_ext, gamma1, gamma2 = self.los_linear_distortions
            gamma1_lenstronomy, gamma2_lenstronomy = ellipticity_slsim_to_lenstronomy(
                e1_slsim=gamma1, e2_slsim=gamma2
            )
            if self.shear:

                kwargs_lens.append(
                    {
                        "gamma1": gamma1_lenstronomy,
                        "gamma2": gamma2_lenstronomy,
                        "ra_0": 0,
                        "dec_0": 0,
                    },
                )
                lens_mass_model_list.append("SHEAR")

            if self.convergence:

                kwargs_lens.append({"kappa": kappa_ext, "ra_0": 0, "dec_0": 0})
                lens_mass_model_list.append("CONVERGENCE")

            self._kwargs_lens = kwargs_lens
            self._lens_mass_model_list = lens_mass_model_list

        else:
            raise ValueError(
                "Deflector model %s not supported for lenstronomy model"
                % self.deflector.deflector_type
            )

        # For significant speedup, use these mass profiles from jaxtronomy

        # TODO: replace with change_source_redshift() currently not fully working
        # self._lens_model.change_source_redshift(z_source=z_source)
        if self.multi_plane:
            lens_redshift_list = self.deflector_redshift
            if self._use_jax is True:
                use_jax = True
            else:
                use_jax = False
        else:
            lens_redshift_list = None
            # For significant speedup, use these mass profiles from jaxtronomy
            if self._use_jax is True:
                use_jax = []
                for profile in self._lens_mass_model_list:
                    if profile in JAX_PROFILES:
                        use_jax.append(True)
                    else:
                        use_jax.append(False)
            else:
                use_jax = False
        lens_model = LensModel(
            lens_model_list=self._lens_mass_model_list,
            cosmo=self.cosmo,
            lens_redshift_list=lens_redshift_list,
            z_lens=self.deflector.redshift,
            z_source=z_source,
            z_source_convention=self.max_redshift_source_class.redshift,
            use_jax=use_jax,
            multi_plane=bool(self.multi_plane),
        )

        return lens_model, self._kwargs_lens

    def deflector_light_model_lenstronomy(self, band):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param band: imaging band
        :type band: str
        :return: lens_light_model_list, kwargs_lens_light
        """
        return self.deflector.light_model_lenstronomy(band=band)

    def source_light_model_lenstronomy(
        self, band=None, time=None, microlensing=False, kwargs_microlensing=None
    ):
        """Returns source light model instance and parameters in lenstronomy
        conventions, which includes extended sources and point sources.

        :param band: imaging band
        :type band: string
        :param time: time is an image observation time in units of days.
            If None, provides magnitude without variability.
        :param microlensing: if using micro-lensing map to produce the
            lensed magnification
        :type microlensing: bool
        :param kwargs_microlensing: additional (optional) dictionary of
            settings required by micro-lensing calculation that do not
            depend on the Lens() class.
        :type kwargs_microlensing: dict
        :return: source_light_model_list, kwargs_source_light
        """
        source_models = {}
        all_source_kwarg_dict = {}

        source_models_list = []
        kwargs_source_list = []
        for index in range(len(self._source)):

            source_model_list, kwargs_source = self.source(index).kwargs_extended_light(
                band=band,
            )
            source_models_list.append(source_model_list)
            kwargs_source_list.append(kwargs_source)
        # lets transform list in to required structure
        source_models_list_restructure = list(np.concatenate(source_models_list))
        kwargs_source_list_restructure = list(np.concatenate(kwargs_source_list))
        source_models["source_light_model_list"] = source_models_list_restructure
        kwargs_source = kwargs_source_list_restructure

        point_source_models_list = []
        kwargs_ps_list = []
        for index in range(len(self._source)):
            ps_type = self.source(index).point_source_type(image_positions=True)
            if ps_type is None:
                pass
            else:
                img_x, img_y = self._point_source_image_positions(
                    source_index=index,
                )
                if band is None:
                    image_magnitudes = np.abs(
                        self._point_source_magnification(source_index=index)
                    )
                else:
                    image_magnitudes = self._point_source_magnitude(
                        band=band,
                        time=time,
                        source_index=index,
                        lensed=True,
                        microlensing=microlensing,
                        kwargs_microlensing=kwargs_microlensing,
                    )
                ps_type, kwargs_ps_ = self.source(index).kwargs_point_source(
                    band, image_pos_x=img_x, image_pos_y=img_y, ps_mag=image_magnitudes
                )
                # TODO: ps_redshift_list to be added
                point_source_models_list.append(ps_type)
                kwargs_ps_list.append(kwargs_ps_)
        source_models["point_source_model_list"] = point_source_models_list
        kwargs_ps = kwargs_ps_list

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
        ra, dec = self.extended_source_image_positions[source_index]

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

        name_def = self.deflector.name
        name_source = self.max_redshift_source_class.name

        return f"{name_def}-{name_source}-LENS_{ra:.4f}_{dec:.4f}"

    def add_subhalos(self, pyhalos_kwargs, dm_type, source_index=0):
        """Generate a realization of the subhalos, halo mass.

        :param pyhalos_kwargs: dictionary of parameters for the pyhalos
            realization.
        :type pyhalos_kwargs: dict
        :param dm_type: type of dark matter models, can be 'CDM', 'WDM',
            or 'ULDM'
        :type dm_type: str
        :param source_index: index of source, default =0, i.e. the first
            source
        :type source_index: int
        """

        z_lens = self.deflector_redshift
        z_source = self.max_redshift_source_class.redshift
        einstein_radius = self._approximate_einstein_radius(source_index)
        cone_opening_angle = 4 * einstein_radius

        if not hasattr(self, "realization"):
            if dm_type == "CDM":
                from pyHalo.PresetModels.cdm import CDM

                realization = CDM(
                    z_lens,
                    z_source,
                    cone_opening_angle_arcsec=cone_opening_angle,
                    **pyhalos_kwargs,
                )
            elif dm_type == "WDM":
                from pyHalo.PresetModels.wdm import WDM

                realization = WDM(
                    z_lens,
                    z_source,
                    cone_opening_angle_arcsec=cone_opening_angle,
                    **pyhalos_kwargs,
                )
            elif dm_type == "ULDM":
                from pyHalo.PresetModels.uldm import ULDM

                realization = ULDM(
                    z_lens,
                    z_source,
                    cone_opening_angle_arcsec=cone_opening_angle,
                    **pyhalos_kwargs,
                )
            else:
                raise ValueError(
                    "We only support 'CDM', 'WDM' or 'ULDM'. "
                    "Received: {}".format(dm_type)
                )

            self.realization = realization
            subhalo_lens_model_list, redshift_array, kwargs_subhalos, _ = (
                self.realization.lensing_quantities(add_mass_sheet_correction=True)
            )
            self._lens_mass_model_list += subhalo_lens_model_list
            self._kwargs_lens += kwargs_subhalos
            print("realization contains " + str(len(realization.halos)) + " halos.")

    def dm_subhalo_mass(self):
        """Get the halo mass of the subhalos in the realization.

        :return: list of halo masses in the realization
        """
        if hasattr(self, "realization"):
            return [halo.mass for halo in self.realization.halos]
        else:
            raise ValueError("No realization found. Please run add_subhalos() first.")

    def subhalos_only_lens_model(self):
        """Get the lens model for the halos only.

        :return: LensModel instance for the halos only, and list of
            kwargs for the subhalos.
        """
        z_lens = self.deflector.redshift
        z_source = self.max_redshift_source_class.redshift
        if hasattr(self, "realization"):
            subhalos_lens_model_list, redshift_array, kwargs_subhalos, _ = (
                self.realization.lensing_quantities(add_mass_sheet_correction=True)
            )
            astropy_instance = self.realization.astropy_instance
        else:
            print("No realization found. Please run add_subhalos() first.")
            kwargs_subhalos = []
            subhalos_lens_model_list = []
            astropy_instance = None
        lens_model_subhalos_only = LensModel(
            lens_model_list=subhalos_lens_model_list,
            cosmo=astropy_instance,
            z_lens=z_lens,
            z_source=z_source,
            z_source_convention=self.max_redshift_source_class.redshift,
            multi_plane=False,
        )
        return lens_model_subhalos_only, kwargs_subhalos

    def lens_to_dataframe(self, index=0, df=None):
        """Store lens properties to a dataframe. This function assumes the name
        of other methods in the lens class. Thus, if the name of some method
        changes, this function will break. Additionally, it assumes that the
        source lives on one plane.

        :param index: index of row that the lens is stored in. Default =
            0
        :type index: int
        :param df: Optional. Stores lens into an existing df if
            necessary, creates one if not.
        :return: pandas DataFrame containing deflector/source mass and
            light properties.
        """
        # TODO: Extend this to work for multiple plane sources
        lens_index = index
        if df is None:
            df = pd.DataFrame()
        # store lens ID
        df.loc[lens_index, "ID"] = str(self.generate_id())

        # store mass model parameters
        for i in self.deflector_mass_model_lenstronomy()[1]:
            for key in i.keys():
                val = i[key]
                df.loc[lens_index, "deflector_mass_" + key] = (
                    safe_value(val)
                    if isinstance(val, (np.ndarray, np.generic, float))
                    else val
                )

        # store light model parameters
        for i in self.deflector_light_model_lenstronomy("i")[1]:
            for key in i.keys():
                val = i[key]
                df.loc[lens_index, "deflector_light_" + key] = safe_value(val)

        # store source light properties
        for i in self.source_light_model_lenstronomy("i")[1]["kwargs_ps"]:
            for key in i.keys():
                if isinstance(i[key], np.ndarray):
                    for j in range(len(i[key])):
                        v = i[key][j]
                        df.loc[lens_index, f"point_source_light_{key}_{j}"] = (
                            safe_value(v)
                        )
        # single float values (velocity dispersion, redshifts)
        df.loc[lens_index, "velocity_dispersion"] = safe_value(
            self.deflector_velocity_dispersion()
        )
        df.loc[lens_index, "deflector_redshift"] = safe_value(self.deflector_redshift)
        df.loc[lens_index, "point_source_redshift"] = safe_value(
            self.source_redshift_list[0]
        )
        ps_times = self.point_source_arrival_times()[0]
        num_images = len(ps_times)
        for i in range(num_images):
            df.loc[lens_index, f"image_{i}_arrival_time"] = ps_times[i]
        df.loc[lens_index, "num_ps_images"] = safe_value(num_images)

        micro_lens_params = (
            self._microlensing_parameters_for_image_positions_single_source(
                band="i", source_index=0
            )
        )
        params = ["kappa_star", "kappa_tot", "shear", "shear_angle"]
        for i, p in enumerate(params):
            for k in range(num_images):
                pls = f"micro_{p}_{k}"
                # check if any of the lists for any param is nested
                param_for_all_images = np.array(micro_lens_params[i])
                if param_for_all_images.shape[0] > 0:
                    param_for_all_images = param_for_all_images.flatten()
                # if param_for_all_images.shape[0]
                val = param_for_all_images[k]
                df.loc[lens_index, pls] = safe_value(val)

        for i in range(num_images):
            df.loc[lens_index, f"point_source_arrival_time_{i}"] = safe_value(
                ps_times[i]
            )
        df.loc[lens_index, "external_shear"] = safe_value(self.external_shear)
        df.loc[lens_index, "extended_unlensed_mag"] = safe_value(
            self.extended_source_magnitude("i", lensed=False)[0]
        )
        df.loc[lens_index, "extended_magnification"] = safe_value(
            self.extended_source_magnification[0]
        )
        return df
