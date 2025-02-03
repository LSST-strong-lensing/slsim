import numpy as np

from slsim.lens import Lens
from typing import Optional
from astropy.cosmology import Cosmology
from slsim.lens import theta_e_when_source_infinity
from slsim.Sources.source_pop_base import SourcePopBase
from slsim.LOS.los_pop import LOSPop
from slsim.Deflectors.deflectors_base import DeflectorsBase
from slsim.lensed_population_base import LensedPopulationBase


class LensPop(LensedPopulationBase):
    """Class to perform samples of lens population."""

    def __init__(
        self,
        deflector_population: DeflectorsBase,
        source_population: SourcePopBase,
        cosmo: Optional[Cosmology] = None,
        sky_area: Optional[float] = None,
        los_pop: Optional[LOSPop] = None,
    ):
        """
        :param deflector_population: Deflector population as an deflectors class
         instance.
        :param source_population: Source population as an sources class inatnce.
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area (solid angle) over which Lens population is sampled.
        :type sky_area: `~astropy.units.Quantity`
        :param los_pop: Configuration for line of sight distribution. Defaults to None.
        :type los_pop: `~LOSPop` or None
        """

        # TODO: ADD EXCEPTION FOR DEFLECTOR AND SOURCE POP FILTER MISMATCH
        super().__init__(sky_area=sky_area, cosmo=cosmo)
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

    def select_lens_at_random(
        self, test_area=None, second_bright_image_cut=None, **kwargs_lens_cut
    ):
        """Draw a random lens within the cuts of the lens and source, with
        possible additional cut in the lensing configuration.

        # TODO: make sure mass function is preserved, # as well as
        option to draw all lenses within the cuts within the area
        :param test_area: solid angle around one lensing galaxies to be
            investigated on (in arc-seconds^2). If None, computed using
            deflector's velocity dispersion.
        :param second_bright_image_cut: Dictionary containing maximum
            magnitude of the second brightest image and corresponding
            band. If provided, selects lenses where the second brightest
            image has a magnitude less than or equal to provided
            magnitude. eg: second_bright_image_cut = {"band": "i",
            "second_bright_mag_max": 23}
        :return: Lens() instance with parameters of the deflector and
            lens and source light
        """
        while True:
            # This creates a single deflector - single_source lens.
            _source = self._sources.draw_source()
            _deflector = self._lens_galaxies.draw_deflector()
            _los = self.los_pop.draw_los(
                source_redshift=_source.redshift, deflector_redshift=_deflector.redshift
            )
            if test_area is None:
                vel_disp = _deflector.velocity_dispersion(cosmo=self.cosmo)
                test_area = draw_test_area(v_sigma=vel_disp)
            else:
                test_area = test_area
            gg_lens = Lens(
                deflector_class=_deflector,
                source_class=_source,
                cosmo=self.cosmo,
                test_area=test_area,
                los_class=_los,
            )
            if gg_lens.validity_test(
                second_bright_image_cut=second_bright_image_cut, **kwargs_lens_cut
            ):
                return gg_lens

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
        second_bright_image_cut=None,
        speed_factor=1,
    ):
        """Return full population list of all lenses within the area.

        # TODO: need to implement a version of it. (improve the
        algorithm)

        :param kwargs_lens_cuts: validity test keywords
        :type kwargs_lens_cuts: dict
        :param multi_source: A boolean value. If True, considers multi
            source lensing. If False, considers single source lensing.
            The default value is True.
        :param second_bright_image_cut: Dictionary containing maximum
            magnitude of the second brightest image and corresponding
            band. If provided, selects lenses where the second brightest
            image has a magnitude less than or equal to provided
            magnitude. eg: second_bright_image_cut = {"band": "i",
            "second_bright_mag_max": 23}
        :param speed_factor: factor by which the number of deflectors is
            decreased to speed up the calculations.
        :return: List of Lens instances with parameters of the
            deflectors and lens and source light.
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
        for _ in range(int(num_lenses / speed_factor)):
            _deflector = self._lens_galaxies.draw_deflector()
            vel_disp = _deflector.velocity_dispersion(cosmo=self.cosmo)
            test_area = draw_test_area(v_sigma=vel_disp)
            num_sources_tested = self.get_num_sources_tested(
                testarea=test_area * speed_factor
            )

            if num_sources_tested > 0:
                valid_sources = []
                n = 0
                while n < num_sources_tested:
                    _source = self._sources.draw_source()
                    if n == 0:
                        # TODO: this is only consistent for a single source. If there
                        # are multiple sources at different redshift, this is not fully
                        # acurate
                        los_class = self.los_pop.draw_los(
                            source_redshift=_source.redshift,
                            deflector_redshift=_deflector.redshift,
                        )
                    lens_class = Lens(
                        deflector_class=_deflector,
                        source_class=_source,
                        cosmo=self.cosmo,
                        test_area=test_area,
                        los_class=los_class,
                    )
                    # Check the validity of the lens system
                    if lens_class.validity_test(
                        second_bright_image_cut=second_bright_image_cut,
                        **kwargs_lens_cuts
                    ):
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
                        test_area=test_area,
                        los_class=los_class,
                    )
                    lens_population.append(lens_final)
        return lens_population


def draw_test_area(**kwargs):
    """Draw a test area around the deflector.

    :param kwargs: Either deflector dictionary or v_sigma for velocity
        dispersion.
    :return: test area in arcsec^2
    """
    theta_e_infinity = theta_e_when_source_infinity(**kwargs)
    test_area = np.pi * (theta_e_infinity * 2.5) ** 2
    return test_area
