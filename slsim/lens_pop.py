import sncosmo
import numpy as np

from slsim.lens import Lens
from typing import Optional, Union
from astropy.cosmology import Cosmology
from slsim.lens import theta_e_when_source_infinity
from slsim.Sources.source_pop_base import SourcePopBase
from slsim.ParamDistributions.los_config import LOSConfig
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
        lightcurve_time: Optional[np.ndarray] = None,
        sn_type: Optional[str] = None,
        sn_absolute_mag_band: Optional[Union[str, sncosmo.Bandpass]] = None,
        sn_absolute_zpsys: Optional[str] = None,
        los_config: Optional[LOSConfig] = None,
        sn_modeldir: Optional[str] = None,
    ):
        """
        Args:
            deflector_population (DeflectorsBase): Deflector population as an instance of a DeflectorsBase subclass.
            source_population (SourcePopBase): Source population as an instance of a SourcePopBase subclass
            cosmo (Optional[Cosmology], optional): AstroPy Cosmology instance. If None, defaults to flat LCDM with h0=0.7 and Om0=0.3.
                                                   Defaults to None.
            lightcurve_time (Optional[np.ndarray], optional): Lightcurve observation time array in units of days. Defaults to None.
            sn_type (Optional[str], optional): Supernova type (Ia, Ib, Ic, IIP, etc.). Defaults to None.
            sn_absolute_mag_band (Optional[Union[str,sncosmo.Bandpass]], optional): Band used to normalize to absolute magnitude.
                                                                                    Defaults to None.
            sn_absolute_zpsys (Optional[str], optional): Zero point system, either AB or Vega, with None defaulting to AB.
                                                         Defaults to None.
            los_config (Optional[LOSConfig], optional): Configuration for line of sight distribution. Defaults to None.
            sn_modeldir (Optional[str], optional): sn_modeldir is the path to the directory containing files needed to initialize
                                                   the sncosmo.model class. For example, sn_modeldir =
                                                   'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can be downloaded
                                                   from https://github.com/LSST-strong-lensing/data_public. For more detail,
                                                   please look at the documentation of RandomizedSupernovae class. Defaults to None.
        """

        # TODO: ADD EXCEPTION FOR DEFLECTOR AND SOURCE POP FILTER MISMATCH
        super().__init__(
            sky_area=sky_area,
            cosmo=cosmo,
            lightcurve_time=lightcurve_time,
            sn_type=sn_type,
            sn_absolute_mag_band=sn_absolute_mag_band,
            sn_absolute_zpsys=sn_absolute_zpsys,
            sn_modeldir=sn_modeldir,
        )
        self.cosmo = cosmo
        self._lens_galaxies = deflector_population
        self._sources = source_population

        self._factor_source = self.sky_area.to_value(
            "deg2"
        ) / self._sources.sky_area.to_value("deg2")
        self._factor_deflector = self.sky_area.to_value(
            "deg2"
        ) / self._lens_galaxies.sky_area.to_value("deg2")
        self.los_config = los_config
        if self.los_config is None:
            self.los_config = LOSConfig()

    def select_lens_at_random(self, test_area=None, **kwargs_lens_cut):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        # TODO: make sure mass function is preserved, # as well as option to draw all
        lenses within the cuts within the area

        :param test_area: area of disk around one lensing galaxies to be investigated on
            (in arc-seconds^2). If None, computed using deflector's velocity dispersion.
        :return: Lens() instance with parameters of the deflector and lens and source
            light
        """
        while True:
            source = self._sources.draw_source()
            lens = self._lens_galaxies.draw_deflector()
            if test_area is None:
                test_area = draw_test_area(deflector=lens)
            else:
                test_area = test_area
            gg_lens = Lens(
                deflector_dict=lens,
                source_dict=source,
                deflector_type=self._lens_galaxies.deflector_profile,
                variability_model=self._sources.variability_model,
                kwargs_variability=self._sources.kwargs_variability,
                sn_type=self.sn_type,
                sn_absolute_mag_band=self.sn_absolute_mag_band,
                sn_absolute_zpsys=self.sn_absolute_zpsys,
                cosmo=self.cosmo,
                test_area=test_area,
                source_type=self._sources.source_type,
                light_profile=self._sources.light_profile,
                lightcurve_time=self.lightcurve_time,
                los_config=self.los_config,
                sn_modeldir=self.sn_modeldir,
                agn_driving_variability_model=self._sources.agn_driving_variability_model,
                agn_driving_kwargs_variability=self._sources.agn_driving_kwargs_variability,
            )
            if gg_lens.validity_test(**kwargs_lens_cut):
                return gg_lens

    @property
    def deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that are being
        considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        return round(self._factor_deflector * self._lens_galaxies.deflector_number())

    @property
    def source_number(self):
        """Number of sources that are being considered to be placed in the sky area
        potentially aligned behind deflectors.

        :return: number of potential sources
        """
        return round(self._factor_source * self._sources.source_number_selected)

    def get_num_sources_tested_mean(self, testarea):
        """Compute the mean of source galaxies needed to be tested within the test area.

        num_sources_tested_mean/ testarea = num_sources/ sky_area; testarea is in units
        of arcsec^2, f_sky is in units of deg^2. 1 deg^2 = 12960000 arcsec^2
        """
        num_sources = self.source_number
        num_sources_tested_mean = (testarea * num_sources) / (
            12960000 * self._factor_source * self._sources.sky_area.to_value("deg2")
        )
        return num_sources_tested_mean

    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """Draw a realization of the expected distribution (Poisson) around the mean for
        the number of source galaxies tested."""
        if num_sources_tested_mean is None:
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    def draw_population(self, kwargs_lens_cuts, speed_factor=1):
        """Return full population list of all lenses within the area # TODO: need to
        implement a version of it. (improve the algorithm)

        :param kwargs_lens_cuts: validity test keywords
        :param speed_factor: factor by which the number of deflectors is decreased to
            speed up the calculations.
        :type kwargs_lens_cuts: dict
        :return: List of Lens instances with parameters of the deflectors and lens and
            source light.
        :rtype: list
        """

        # Initialize an empty list to store the Lens instances
        gg_lens_population = []
        # Estimate the number of lensing systems
        num_lenses = self.deflector_number
        # num_sources = self._source_galaxies.galaxies_number()
        #        print(num_sources_tested_mean)
        #        print("num_lenses is " + str(num_lenses))
        #        print("num_sources is " + str(num_sources))
        #        print(np.int(num_lenses * num_sources_tested_mean))

        # Draw a population of galaxy-galaxy lenses within the area.
        for _ in range(int(num_lenses / speed_factor)):
            lens = self._lens_galaxies.draw_deflector()
            test_area = draw_test_area(deflector=lens)
            num_sources_tested = self.get_num_sources_tested(
                testarea=test_area * speed_factor
            )
            # TODO: to implement this for a multi-source plane lens system
            if num_sources_tested > 0:
                n = 0
                while n < num_sources_tested:
                    source = self._sources.draw_source()
                    gg_lens = Lens(
                        deflector_dict=lens,
                        source_dict=source,
                        deflector_type=self._lens_galaxies.deflector_profile,
                        variability_model=self._sources.variability_model,
                        kwargs_variability=self._sources.kwargs_variability,
                        sn_type=self.sn_type,
                        sn_absolute_mag_band=self.sn_absolute_mag_band,
                        sn_absolute_zpsys=self.sn_absolute_zpsys,
                        cosmo=self.cosmo,
                        test_area=test_area,
                        source_type=self._sources.source_type,
                        los_config=self.los_config,
                        light_profile=self._sources.light_profile,
                        lightcurve_time=self.lightcurve_time,
                        sn_modeldir=self.sn_modeldir,
                        agn_driving_variability_model=self._sources.agn_driving_variability_model,
                        agn_driving_kwargs_variability=self._sources.agn_driving_kwargs_variability,
                    )
                    # Check the validity of the lens system
                    if gg_lens.validity_test(**kwargs_lens_cuts):
                        gg_lens_population.append(gg_lens)
                        # if a lens system passes the validity test, code should exit
                        # the loop. so, n should be greater or equal to
                        # num_sources_tested which will break the while loop
                        # (instead of this one can simply use break).
                        n = num_sources_tested
                    else:
                        n += 1
        return gg_lens_population


def draw_test_area(deflector):
    """Draw a test area around the deflector.

    :param deflector: deflector dictionary
    :return: test area in arcsec^2
    """
    theta_e_infinity = theta_e_when_source_infinity(deflector)
    test_area = np.pi * (theta_e_infinity * 2.5) ** 2
    return test_area
