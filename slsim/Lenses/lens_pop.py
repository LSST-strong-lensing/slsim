import numpy as np

from slsim.Lenses.lens import Lens
from typing import Optional
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from slsim.Sources.SourcePopulation.source_pop_base import SourcePopBase
from slsim.LOS.los_pop import LOSPop
from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase
from slsim.Lenses.lensed_population_base import LensedPopulationBase

from tqdm import tqdm


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
                                    "mag_arc_limit": {"i", 24},
                                    "second_brightest_image_cut": {"i", 24}
                                }. All these cuts are optional.
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
            _deflector.update_center(deflector_area=0.01)
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

        # Initialize an empty list to store the Lens instances
        lens_population = []
        # Estimate the number of lensing systems
        num_lenses = self.deflector_number
        # num_sources = self._source_galaxies.galaxies_number()
        #        print(num_sources_tested_mean)
        #        print("num_lenses is " + str(num_lenses))
        #        print("num_sources is " + str(num_sources))
        #        print(np.int(num_lenses * num_sources_tested_mean))

        # Draw a population of galaxy-galaxy lenses within the area.
        for _ in tqdm(
            range(int(num_lenses / speed_factor)),
            disable=not verbose,
            desc="Drawing lens population",
        ):
            _deflector = self._lens_galaxies.draw_deflector()
            _deflector.update_center(deflector_area=0.01)
            theta_e_infinity = _deflector.theta_e_infinity(cosmo=self.cosmo)
            test_area = area_theta_e_infinity(theta_e_infinity=theta_e_infinity)
            num_sources_tested = self.get_num_sources_tested(
                testarea=test_area * speed_factor
            )

            if num_sources_tested > 0:
                valid_sources = []
                n = 0
                while n < num_sources_tested:
                    _source = self._sources.draw_source()
                    _source.update_center(
                        area=test_area, reference_position=_deflector.deflector_center
                    )
                    if n == 0:
                        # TODO: this is only consistent for a single source. If there
                        #  are multiple sources at different redshift, this is not fully
                        #  accurate
                        los_class = self.los_pop.draw_los(
                            source_redshift=_source.redshift,
                            deflector_redshift=_deflector.redshift,
                        )
                    lens_class = Lens(
                        deflector_class=_deflector,
                        source_class=_source,
                        cosmo=self.cosmo,
                        los_class=los_class,
                        use_jax=self._use_jax,
                    )
                    # Check the validity of the lens system
                    if lens_class.validity_test(**kwargs_lens_cuts):
                        valid_sources.append(_source)
                        # If multi_source is False, stop after finding the first valid source
                        if not multi_source:
                            break
                    n += 1
                if len(valid_sources) > 0:
                    # Use a single source if only one source is valid, else use
                    # the list of valid sources
                    if len(valid_sources) == 1:
                        final_sources = valid_sources[0]
                    else:
                        final_sources = valid_sources
                    lens_final = Lens(
                        deflector_class=_deflector,
                        source_class=final_sources,
                        cosmo=self.cosmo,
                        los_class=los_class,
                        use_jax=self._use_jax,
                    )
                    lens_population.append(lens_final)
        return lens_population


def area_theta_e_infinity(theta_e_infinity):
    """Draw a test area around the deflector.

    :param theta_e_infinity: Einstein radius for infinitly far away
        source (Dds/Ds = 1)
    :return: test area in arcsec^2
    """
    test_area = np.pi * (theta_e_infinity * 1.5) ** 2
    return test_area
