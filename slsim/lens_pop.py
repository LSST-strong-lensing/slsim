from slsim.ParamDistributions.los_config import LOSConfig
import os
import pickle

import numpy as np
from astropy.table import Table

from slsim.lens import Lens
from slsim.lens import theta_e_when_source_infinity
from slsim.lensed_population_base import LensedPopulationBase
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from slsim.Deflectors.deflectors_base import DeflectorsBase
from slsim.Sources.source_pop_base import SourcePopBase


class LensPop(LensedPopulationBase):
    """Class to perform samples of lens population."""

    def __init__(
        self,
        deflector_population: DeflectorsBase,
        source_population: SourcePopBase,
        deflector_type="elliptical",
        source_type="galaxies",
        kwargs_deflector_cut=None,
        kwargs_source_cut=None,
        kwargs_quasars=None,
        kwargs_quasars_galaxies=None,
        variability_model=None,
        kwargs_variability=None,
        kwargs_mass2light=None,
        skypy_config=None,
        slhammocks_config=None,
        sky_area=None,
        source_sky_area=None,
        deflector_sky_area=None,
        filters=None,
        cosmo=None,
        source_light_profile="single_sersic",
        catalog_type="skypy",
        catalog_path=None,
        lightcurve_time=None,
        sn_type=None,
        sn_absolute_mag_band=None,
        sn_absolute_zpsys=None,
        los_config=None,
        sn_modeldir=None,
    ):
        """

        :param deflector_type: type of the lens
        :type deflector_type: string
        :param source_type: type of the source
        :type source_type: string
        :param kwargs_deflector_cut: cuts on the deflector to be excluded in the sample
        :type kwargs_deflector_cut: dict
        :param kwargs_source_cut: cuts on the source to be excluded in the sample
        :type kwargs_source_cut: dict
        :param kwargs_quasars: a dict of keyword arguments which is an input for
         quasar_catalog. Please look at quasar_catalog/simple_quasar.py.
        :param variability_model: keyword for variability model to be used. This is an
         input for the Variability class.
        :type variability_model: str
        :param kwargs_variability: keyword arguments for the variability of a source.
         This is associated with an input for Variability class.
        :type kwargs_variability: list of str
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string
        :param slhammocks_config: path to the deflector population csv file for 'halo-model'
        :type slhammocks_config: string
        :param sky_area: Sky area over which lens population will be simulated. If
         sky_area is not None, number of source sample and deflector sample within a
         source_sky_area and deflector_sky_area will be scaled to the sky_area.
         This will allow us to simulate lens population over a large sky area without
         further significant computational cost.
        :param source_sky_area: Sky area over which sources are sampled. Must be in
         units of solid angle. If None, source_sky_area will be equal to sky_area.
        :type source_sky_area: `~astropy.units.Quantity`
        :param deflector_sky_area: Sky area over which deflectors are sampled. Must be
         in units of solid angle. If None, deflcetor_sky_area will be equal to sky_area.
        :type deflector_sky_area: `~astropy.units.Quantity`
        :type sky_area: `~astropy.units.Quantity`
        :param filters: filters for SED integration
        :type filters: list of strings or None
        :param cosmo: cosmology object
        :type cosmo: `~astropy.cosmology.FLRW`
        :param source_light_profile: keyword for number of sersic profile to use in
         source light model. It is necessary to recognize quantities given in the source
         catalog.
        :type source_light_profile: str . Either "single" or "double" .
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it.
        :type catalog_type: str. eg: "scotch"
        :param catalog_path: path to the source catalog. If None, existing source
         catalog within the slsim will be used. We have used small subset of scotch
         catalog. So, if one wants to use full scotch catalog, they can set path to
         their path to local drive.
        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array
        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param sn_absolute_mag_band: Band used to normalize to absolute magnitude
        :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
        :param sn_absolute_zpsys: Optional, AB or Vega (AB default)
        :type sn_absolute_zpsys: str
        :param los_config: configuration for line of sight distribution
        :type los_config: LOSConfig instance
        :param sn_modeldir: sn_modeldir is the path to the directory containing files
         needed to initialize the sncosmo.model class. For example,
         sn_modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can
         be downloaded from https://github.com/LSST-strong-lensing/data_public .
         For more detail, please look at the documentation of RandomizedSupernovae
         class.
        :type sn_modeldir: str
        """
        super().__init__(
            sky_area,
            cosmo,
            lightcurve_time,
            sn_type,
            sn_absolute_mag_band,
            sn_absolute_zpsys,
            sn_modeldir,
        )
        self.cosmo = cosmo
        self._lens_galaxies = deflector_population
        self._sources = source_population

        self._factor_source = self.f_sky.to_value(
            "deg2"
        ) / self._sources._sky_area.to_value("deg2")
        self._factor_deflector = self.f_sky.to_value(
            "deg2"
        ) / self._lens_galaxies._sky_area.to_value("deg2")
        self.los_config = los_config
        if self.los_config is None:
            self.los_config = LOSConfig()

    def select_lens_at_random(self, **kwargs_lens_cut):
        """Draw a random lens within the cuts of the lens and source, with possible
        additional cut in the lensing configuration.

        # TODO: make sure mass function is preserved, # as well as option to draw all
        lenses within the cuts within the area

        :return: Lens() instance with parameters of the deflector and lens and source
            light
        """
        while True:
            source = self._sources.draw_source()
            lens = self._lens_galaxies.draw_deflector()
            gg_lens = Lens(
                deflector_dict=lens,
                source_dict=source,
                variability_model=self._sources.variability_model,
                kwargs_variability=self._sources.kwargs_variability,
                sn_type=self.sn_type,
                sn_absolute_mag_band=self.sn_absolute_mag_band,
                sn_absolute_zpsys=self.sn_absolute_zpsys,
                cosmo=self.cosmo,
                source_type=self._source_model_type,
                light_profile=self._sources.light_profile,
                lightcurve_time=self.lightcurve_time,
                los_config=self.los_config,
                sn_modeldir=self.sn_modeldir,
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

        num_sources_tested_mean/ testarea = num_sources/ f_sky; testarea is in units of
        arcsec^2, f_sky is in units of deg^2. 1 deg^2 = 12960000 arcsec^2
        """
        num_sources = self.source_number
        num_sources_tested_mean = (testarea * num_sources) / (
            12960000 * self._factor_source * self.source_sky_area.to_value("deg2")
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
                        variability_model=self._sources.variability_model,
                        kwargs_variability=self._sources.kwargs_variability,
                        sn_type=self.sn_type,
                        sn_absolute_mag_band=self.sn_absolute_mag_band,
                        sn_absolute_zpsys=self.sn_absolute_zpsys,
                        cosmo=self.cosmo,
                        test_area=test_area,
                        source_type=self._source_model_type,
                        los_config=self.los_config,
                        light_profile=self._sources.light_profile,
                        lightcurve_time=self.lightcurve_time,
                        sn_modeldir=self.sn_modeldir,
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
