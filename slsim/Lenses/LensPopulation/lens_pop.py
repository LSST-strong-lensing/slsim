import numpy as np
from astropy.table import Table
from tqdm import tqdm

from slsim.Lenses.lens import Lens
from typing import Optional
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from slsim.Sources.SourcePopulation.source_pop_base import SourcePopBase
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Sources.SourcePopulation.point_sources import PointSources
from slsim.Sources.SourcePopulation.point_plus_extended_sources import (
    PointPlusExtendedSources,
)
from slsim.LOS.los_pop import LOSPop
from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase
from slsim.Lenses.LensPopulation.lensed_population_base import LensedPopulationBase
from slsim.Util.cosmo_util import get_cosmo_cache


class LensPop(LensedPopulationBase):
    """Class to perform samples of lens population."""

    def __init__(
        self,
        deflector_population: DeflectorsBase,
        source_population: SourcePopBase,
        cosmo: Optional[Cosmology] = None,
        sky_area: Optional[float or Quantity] = None,
        los_pop: Optional[LOSPop] = None,
        use_jax=True,
    ):
        """
        :param deflector_population: Deflector population as a deflectors class
         instance.
        :param source_population: Source population as a sources class inatnce.
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area (solid angle) over which Lens population is sampled.
        :type sky_area: `~astropy.units.Quantity`
        :param los_pop: Configuration for line of sight distribution. Defaults to None.
        :type los_pop: `~LOSPop` or None
        :param use_jax: if True, will use JAX version of lenstronomy to do lensing calculations for models that are
         supported in JAXtronomy
        :type use_jax: bool
        """

        # TODO: ADD EXCEPTION FOR DEFLECTOR AND SOURCE POP FILTER MISMATCH
        super().__init__(sky_area=sky_area, cosmo=cosmo, use_jax=use_jax)
        self.cosmo = cosmo
        self._lens_galaxies = deflector_population
        self._sources = source_population
        self._factor_source = self.sky_area.to_value(
            "deg2"
        ) / self._sources.sky_area.to_value("deg2")
        self._factor_deflector = self.sky_area.to_value(
            "deg2"
        ) / self._lens_galaxies.sky_area.to_value("deg2")
        self.los_pop = los_pop
        if self.los_pop is None:
            self.los_pop = LOSPop()

    def select_lens_at_random(self, test_area=None, verbose=False, **kwargs_lens_cut):
        """Draw a random lens within the cuts of the lens and source, with
        possible additional cuts in the lensing configuration.

        # TODO: Make sure mass function is preserved, as well as the option to draw all lenses
        within the cuts within the area.

        :param test_area: Solid angle around one lensing galaxy to be investigated on
                        (in arc-seconds^2). If None, computed using deflector's velocity dispersion.
        :type test_area: float or None
        :param kwargs_lens_cut: Dictionary of cuts that one wants to apply to the lens.
                                Example: kwargs_lens_cut = {
                                "min_image_separation": 0.5,
                                "max_image_separation": 10,
                                "mag_arc_limit": {"i": 24},
                                "second_brightest_image_cut": {"i": 24}}.
                                All these cuts are optional.
        :type kwargs_lens_cut: dict
        :param verbose: print statements added
        :type verbose: bool
        :return: Lens() instance with parameters of the deflector and lens and source light.
        :rtype: Lens
        """
        n = 0
        while True:
            # This creates a single deflector - single_source lens.
            _source = self._draw_source(**kwargs_lens_cut)
            _deflector = self._lens_galaxies.draw_deflector()
            _los = self.los_pop.draw_los(
                source_redshift=_source.redshift, deflector_redshift=_deflector.redshift
            )
            if test_area is None:
                theta_e_infinity = _deflector.theta_e_infinity(cosmo=self.cosmo)
                test_area_ = area_theta_e_infinity(theta_e_infinity=theta_e_infinity)
            else:
                test_area_ = test_area
            # set a center for the deflector and source
            _deflector.update_center(area=0.01)
            _source.update_center(
                area=test_area_, reference_position=_deflector.deflector_center
            )
            gg_lens = Lens(
                deflector_class=_deflector,
                source_class=_source,
                cosmo=self.cosmo,
                los_class=_los,
                use_jax=self._use_jax,
            )
            if gg_lens.validity_test(**kwargs_lens_cut):
                if verbose is True:
                    print("selected lens after %s tries." % n)
                return gg_lens
            n += 1

    def _draw_source(self, mag_arc_limit=None, magnification_limit=2, **kwargs):
        """Draw from source population considering some additional constraints
        to be fulfilled.

        In particular, we are using a maximal intrinsic source magnitude
        2 magnitudes fainter than the limit of a detectable lensed arc.

        :param mag_arc_limit: dictionary with key of bands and values of
            magnitude limits of integrated lensed arc
        :type mag_arc_limit: dict with key of bands and values of
            magnitude limits
        :param magnification_limit: lensing magnification limit that
            intrinsic sources fainter than mag_source > mag_arc_limit +
            magnification_limit are ignored
        :type magnification_limit: float >=1
        :param kwargs: additional Lens.validity_test() arguments that
            are not used
        :return: Source() class that approximately satisfies additional
            selection
        """
        _source = self._sources.draw_source()
        if mag_arc_limit is None:
            return self._sources.draw_source()
        n = 0
        while True and n < 1000:
            _source = self._sources.draw_source()
            condition = True
            for band, mag_limit_band in mag_arc_limit.items():
                mag_source = _source.extended_source_magnitude(band)
                if mag_source > mag_limit_band + magnification_limit:
                    condition = False
            if condition is True:
                return _source
            else:
                n += 1
        raise ValueError(
            "selecting a source to match the mag_arc_limit %s did not work with %s tries with a magnification cut at %s."
            % (mag_arc_limit, 1000, magnification_limit)
        )

    @property
    def deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that
        are being considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        return round(self._factor_deflector * self._lens_galaxies.deflector_number())

    @property
    def source_number(self):
        """Number of sources that are being considered to be placed in the sky
        area potentially aligned behind deflectors.

        :return: number of potential sources
        """
        return round(self._factor_source * self._sources.source_number_selected)

    def get_num_sources_tested_mean(self, testarea):
        """Compute the mean of source galaxies needed to be tested within the
        test area.

        num_sources_tested_mean/ testarea = num_sources/ sky_area;
        testarea is in units of arcsec^2, f_sky is in units of deg^2. 1
        deg^2 = 12960000 arcsec^2
        """
        num_sources = self.source_number
        num_sources_tested_mean = (testarea * num_sources) / (
            12960000 * self._factor_source * self._sources.sky_area.to_value("deg2")
        )
        return num_sources_tested_mean

    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """Draw a realization of the expected distribution (Poisson) around the
        mean for the number of source galaxies tested."""
        if num_sources_tested_mean is None:
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    def draw_population(
        self,
        kwargs_lens_cuts,
        multi_source=False,
        speed_factor=1,
        verbose=False,
    ):
        """Return full population list of all lenses within the area.

        # TODO: need to implement a version of it. (improve the
        algorithm)

        :param kwargs_lens_cuts: validity test keywords. dictionary of
            cuts that one wants to apply to the lens. eg:
            kwargs_lens_cut = {}"min_image_separation": 0.5,
            "max_image_separation": 10, "mag_arc_limit": {"i", 24},
            "second_bright_image_cut = {"band": ["i"], "mag_max":[23]}.
            all these cuts are optional.
        :type kwargs_lens_cuts: dict
        :param multi_source: A boolean value. If True, considers multi
            source lensing. If False, considers single source lensing.
            The default value is True.
        :param speed_factor: factor by which the number of deflectors is
            decreased to speed up the calculations.
        :return: List of Lens instances with parameters of the
            deflectors and lens and source light.
        :param verbose: If True, prints progress information. Default is
            False.
        :type verbose: bool
        :rtype: list
        """

        num_lenses = self.deflector_number
        num_draws = int(num_lenses / speed_factor)
        if num_draws <= 0:
            return []

        fast_population = self._try_draw_population_vectorized(
            num_draws=num_draws,
            kwargs_lens_cuts=kwargs_lens_cuts,
            multi_source=multi_source,
            speed_factor=speed_factor,
            verbose=verbose,
        )
        if fast_population is not None:
            return fast_population

        lens_population = []
        # Draw a population of galaxy-galaxy lenses within the area.
        for _ in tqdm(
            range(num_draws),
            disable=not verbose,
            desc="Drawing lens population",
        ):
            _deflector = self._lens_galaxies.draw_deflector()
            _deflector.update_center(area=0.01)
            theta_e_infinity = _deflector.theta_e_infinity(
                cosmo=self.cosmo, use_jax=self._use_jax
            )
            test_area = area_theta_e_infinity(theta_e_infinity=theta_e_infinity)
            num_sources_tested = self.get_num_sources_tested(
                testarea=test_area * speed_factor
            )
            lens_final = self._draw_sources_unscreened(
                deflector=_deflector,
                test_area=test_area,
                num_sources_tested=num_sources_tested,
                kwargs_lens_cuts=kwargs_lens_cuts,
                multi_source=multi_source,
            )
            if lens_final is not None:
                lens_population.append(lens_final)
        return lens_population

    def _build_lens_from_sources(self, deflector, valid_sources, los_class):
        """Construct the final ``Lens`` object from surviving sources."""
        final_sources = valid_sources[0] if len(valid_sources) == 1 else valid_sources
        return Lens(
            deflector_class=deflector,
            source_class=final_sources,
            cosmo=self.cosmo,
            los_class=los_class,
            use_jax=self._use_jax,
        )

    def _draw_sources_unscreened(
        self,
        deflector,
        test_area,
        num_sources_tested,
        kwargs_lens_cuts,
        multi_source,
    ):
        """Draw and validate sources sequentially for a single deflector."""
        if num_sources_tested <= 0:
            return None
        valid_sources = []
        los_class = None
        for n in range(num_sources_tested):
            _source = self._sources.draw_source()
            _source.update_center(
                area=test_area, reference_position=deflector.deflector_center
            )
            if n == 0:
                los_class = self.los_pop.draw_los(
                    source_redshift=_source.redshift,
                    deflector_redshift=deflector.redshift,
                )
            lens_class = Lens(
                deflector_class=deflector,
                source_class=_source,
                cosmo=self.cosmo,
                los_class=los_class,
                use_jax=self._use_jax,
            )
            if lens_class.validity_test(**kwargs_lens_cuts):
                valid_sources.append(_source)
                if not multi_source:
                    break
        if len(valid_sources) == 0:
            return None
        return self._build_lens_from_sources(deflector, valid_sources, los_class)

    def _is_source_population_vectorizable(self):
        """Return True when ``self._sources`` supports table-indexed draws."""
        src = self._sources
        if not isinstance(getattr(src, "_objects_select", None), Table):
            return False
        if type(src).draw_object is not SourcePopBase.draw_object:
            return False
        for base in (Galaxies, PointPlusExtendedSources, PointSources):
            if isinstance(src, base) and type(src).draw_source is base.draw_source:
                return True
        return False

    def _extract_source_screen_arrays(self):
        """Extract vectorized source redshifts and point-source offsets."""
        src_table = self._sources._objects_select
        src_z = np.asarray(src_table["z"], dtype=float)
        if not np.all(np.isfinite(src_z)):
            return None
        has_ps = getattr(self._sources, "_point_source_type", None) is not None
        if not has_ps:
            return src_z, None, None
        joint_kw = getattr(self._sources, "_joint_point_source_kwargs", None) or {}
        if "ra_off" in src_table.colnames and "dec_off" in src_table.colnames:
            ra_off = np.asarray(src_table["ra_off"], dtype=float)
            dec_off = np.asarray(src_table["dec_off"], dtype=float)
            if not (np.all(np.isfinite(ra_off)) and np.all(np.isfinite(dec_off))):
                return None
            return src_z, ra_off, dec_off
        ra_default = float(joint_kw.get("ra_off", 0.0))
        dec_default = float(joint_kw.get("dec_off", 0.0))
        if ra_default == 0.0 and dec_default == 0.0:
            return src_z, None, None
        return (
            src_z,
            np.full(len(src_z), ra_default, dtype=float),
            np.full(len(src_z), dec_default, dtype=float),
        )

    @staticmethod
    def _screen_candidates_for_deflector(
        z_d,
        theta_e_inf,
        n_test,
        src_z,
        src_ra_off,
        src_dec_off,
        cosmo_cache,
        use_linear_bound,
        dds_ds_100,
    ):
        """Sample source draws in the deflector test area and filter
        geometrically."""
        cand_idx = np.random.randint(0, len(src_z), size=n_test)
        u = np.random.random(n_test)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=n_test)
        radii = 1.5 * theta_e_inf * np.sqrt(u)
        dx = radii * np.cos(phi)
        dy = radii * np.sin(phi)
        if src_ra_off is not None:
            ps_dx = dx + src_ra_off[cand_idx]
            ps_dy = dy + src_dec_off[cand_idx]
            beta_sq = ps_dx * ps_dx + ps_dy * ps_dy
        else:
            beta_sq = dx * dx + dy * dy

        z_s = src_z[cand_idx]
        z_pass = np.flatnonzero(z_s > z_d)
        if z_pass.size == 0:
            return cand_idx, dx, dy, z_pass

        dds_ds = cosmo_cache.dds_over_ds(
            np.full(z_pass.size, z_d, dtype=float), z_s[z_pass]
        )
        if use_linear_bound:
            theta_e_bound = 2.5 * theta_e_inf * np.maximum(dds_ds, 0.0)
        else:
            # For any non-negative monotonic density profile (e.g., NFW+Hernquist),
            # theta_E(z_s) <= sqrt((D_ds/D_s)(z_s) / (D_ds/D_s)(100)) * theta_E_inf.
            ratio = np.clip(dds_ds / max(float(dds_ds_100), 1e-12), 0.0, 1.0)
            theta_e_bound = 2.5 * theta_e_inf * np.sqrt(ratio)
        keep = z_pass[beta_sq[z_pass] <= theta_e_bound * theta_e_bound]
        return cand_idx, dx, dy, keep

    def _try_draw_population_vectorized(
        self,
        num_draws,
        kwargs_lens_cuts,
        multi_source,
        speed_factor,
        verbose,
    ):
        """Vectorized population screening and lazy object construction.

        Returns None when the deflector or source population requires
        the generic per-object fallback loop.
        """
        if type(self).get_num_sources_tested is not LensPop.get_num_sources_tested:
            return None
        if (
            type(self).get_num_sources_tested_mean
            is not LensPop.get_num_sources_tested_mean
        ):
            return None
        if (
            type(self._lens_galaxies).draw_deflector
            is not DeflectorsBase.draw_deflector
            or type(self._lens_galaxies).draw_object is not SourcePopBase.draw_object
        ):
            return None
        n_def_sel = self._lens_galaxies.deflector_number()
        n_src_sel = self._sources.source_number_selected
        if n_def_sel <= 0 or n_src_sel <= 0:
            return None
        if not self._is_source_population_vectorizable():
            return None
        src_arrays = self._extract_source_screen_arrays()
        if src_arrays is None:
            return None
        src_z, src_ra_off, src_dec_off = src_arrays

        def_idx = np.random.randint(0, n_def_sel, size=num_draws)
        theta_e_inf = self._lens_galaxies._theta_e_infinity_for_selected(
            def_idx, cosmo=self.cosmo
        )
        if theta_e_inf is None:
            return None

        test_areas = area_theta_e_infinity(theta_e_infinity=theta_e_inf)
        means = np.maximum(
            self.get_num_sources_tested_mean(test_areas * speed_factor), 0.0
        )
        num_sources_tested = np.random.poisson(lam=means)
        cand_def_positions = np.flatnonzero(num_sources_tested > 0)
        if cand_def_positions.size == 0:
            return []

        cosmo_cache = get_cosmo_cache(self.cosmo)
        def_z_all = np.asarray(
            self._lens_galaxies._objects_select["z"][def_idx[cand_def_positions]],
            dtype=float,
        )
        use_linear_bound = self._lens_galaxies.mass_type == "EPL" and (
            getattr(self._lens_galaxies, "_sis_convention", True)
            or getattr(self._lens_galaxies, "_gamma_pl", None) in (None, 2, 2.0)
        )
        dds_ds_100_all = cosmo_cache.dds_over_ds(
            def_z_all, np.full(def_z_all.size, 100.0, dtype=float)
        )

        lens_population = []
        for idx_pos, def_pos in tqdm(
            enumerate(cand_def_positions),
            total=int(cand_def_positions.size),
            disable=not verbose,
            desc="Drawing lens population",
        ):
            n_test = int(num_sources_tested[def_pos])
            cand_idx, dx, dy, keep = self._screen_candidates_for_deflector(
                z_d=float(def_z_all[idx_pos]),
                theta_e_inf=float(theta_e_inf[def_pos]),
                n_test=n_test,
                src_z=src_z,
                src_ra_off=src_ra_off,
                src_dec_off=src_dec_off,
                cosmo_cache=cosmo_cache,
                use_linear_bound=use_linear_bound,
                dds_ds_100=float(dds_ds_100_all[idx_pos]),
            )
            if keep.size == 0:
                continue

            _deflector = self._lens_galaxies._draw_deflector_at_selected(
                int(def_idx[def_pos])
            )
            _deflector.update_center(area=0.01)
            def_center = _deflector.deflector_center
            los_class = self.los_pop.draw_los(
                source_redshift=float(src_z[cand_idx[0]]),
                deflector_redshift=_deflector.redshift,
            )

            valid_sources = []
            for pos in keep:
                _source = self._sources._draw_source_at_selected(int(cand_idx[pos]))
                _source.update_center(
                    center_x=def_center[0] + dx[pos],
                    center_y=def_center[1] + dy[pos],
                )
                lens_class = Lens(
                    deflector_class=_deflector,
                    source_class=_source,
                    cosmo=self.cosmo,
                    los_class=los_class,
                    use_jax=self._use_jax,
                )
                if lens_class.validity_test(**kwargs_lens_cuts):
                    valid_sources.append(_source)
                    if not multi_source:
                        break
            if len(valid_sources) > 0:
                lens_population.append(
                    self._build_lens_from_sources(_deflector, valid_sources, los_class)
                )
        return lens_population


def area_theta_e_infinity(theta_e_infinity):
    """Draw a test area around the deflector.

    :param theta_e_infinity: Einstein radius for infinitly far away
        source (Dds/Ds = 1)
    :return: test area in arcsec^2
    """
    test_area = np.pi * (theta_e_infinity * 1.5) ** 2
    return test_area
