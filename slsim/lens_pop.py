from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from slsim.lens import (
    Lens,
    theta_e_when_source_infinity,
)
import numpy as np
from slsim.lensed_population_base import LensedPopulationBase
import os
import pickle
from astropy.table import Table


class LensPop(LensedPopulationBase):
    """Class to perform samples of lens population."""

    def __init__(
        self,
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
        sky_area=None,
        filters=None,
        cosmo=None,
        source_light_profile="single_sersic",
        catalog_type="skypy",
        catalog_path=None,
        lightcurve_time=None,
        sn_type=None,
        sn_absolute_mag_band=None,
        sn_absolute_zpsys=None,
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
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param filters: filters for SED integration
        :type filters: list of strings or None
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
        """
        super().__init__(
            sky_area,
            cosmo,
            lightcurve_time,
            sn_type,
            sn_absolute_mag_band,
            sn_absolute_zpsys,
        )
        if source_type == "galaxies" and kwargs_variability is not None:
            raise ValueError(
                "Galaxies cannot have variability. Either choose"
                "point source (eg: quasars) or do not provide kwargs_variability."
            )

        if deflector_type in ["elliptical", "all-galaxies"] or source_type in [
            "galaxies"
        ]:
            pipeline = SkyPyPipeline(
                skypy_config=skypy_config,
                sky_area=sky_area,
                filters=filters,
                cosmo=cosmo,
            )
        if kwargs_deflector_cut is None:
            kwargs_deflector_cut = {}
        if kwargs_mass2light is None:
            kwargs_mass2light = {}

        if deflector_type == "elliptical":
            from slsim.Deflectors.elliptical_lens_galaxies import (
                EllipticalLensGalaxies,
            )

            self._lens_galaxies = EllipticalLensGalaxies(
                pipeline.red_galaxies,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

        elif deflector_type == "all-galaxies":
            from slsim.Deflectors.all_lens_galaxies import AllLensGalaxies

            red_galaxy_list = pipeline.red_galaxies
            blue_galaxy_list = pipeline.blue_galaxies

            self._lens_galaxies = AllLensGalaxies(
                red_galaxy_list=red_galaxy_list,
                blue_galaxy_list=blue_galaxy_list,
                kwargs_cut=kwargs_deflector_cut,
                kwargs_mass2light=kwargs_mass2light,
                cosmo=cosmo,
                sky_area=sky_area,
            )

        else:
            raise ValueError("deflector_type %s is not supported" % deflector_type)

        if kwargs_source_cut is None:
            kwargs_source_cut = {}
        if source_type == "galaxies":
            from slsim.Sources.galaxies import Galaxies

            self._sources = Galaxies(
                pipeline.blue_galaxies,
                kwargs_cut=kwargs_source_cut,
                cosmo=cosmo,
                sky_area=sky_area,
                light_profile=source_light_profile,
                catalog_type=catalog_type,
            )
            self._source_model_type = "extended"
        elif source_type == "quasars":
            from slsim.Sources.point_sources import PointSources
            from slsim.Sources.QuasarCatalog.simple_quasar import quasar_catalog_simple

            if kwargs_quasars is None:
                kwargs_quasars = {}
            quasar_source = quasar_catalog_simple(**kwargs_quasars)
            self._sources = PointSources(
                quasar_source,
                cosmo=cosmo,
                sky_area=sky_area,
                variability_model=variability_model,
                kwargs_variability_model=kwargs_variability,
                light_profile=source_light_profile,
            )
            self._source_model_type = "point_source"
        elif source_type == "quasar_plus_galaxies":
            from slsim.Sources.point_plus_extended_sources import (
                PointPlusExtendedSources,
            )
            from slsim.Sources.QuasarCatalog.quasar_plus_galaxies import (
                quasar_galaxies_simple,
            )

            if kwargs_quasars_galaxies is None:
                kwargs_quasars_galaxies = {}
            quasar_galaxy_source = quasar_galaxies_simple(**kwargs_quasars_galaxies)
            self._sources = PointPlusExtendedSources(
                quasar_galaxy_source,
                cosmo=cosmo,
                sky_area=sky_area,
                kwargs_cut=kwargs_source_cut,
                variability_model=variability_model,
                kwargs_variability_model=kwargs_variability,
                light_profile=source_light_profile,
                catalog_type=catalog_type,
            )
            self._source_model_type = "point_plus_extended"
        elif source_type == "supernovae_plus_galaxies":
            from slsim.Sources.point_plus_extended_sources import (
                PointPlusExtendedSources,
            )

            # currently, we are using precomputed supernovae catlog. Future plan is to
            # develop a supernovae class inside the slsim and them here to generate
            # supernovae light curves.
            self.path = os.path.dirname(__file__)
            if catalog_type == "scotch":
                if catalog_path is not None:
                    new_path = catalog_path
                else:
                    new_path = (
                        self.path + "/Sources/SupernovaeData/scotch_host_data.fits"
                    )
                load_supernovae_data = Table.read(
                    new_path,
                    format="fits",
                )
                self._sources = PointPlusExtendedSources(
                    load_supernovae_data,
                    cosmo=cosmo,
                    sky_area=sky_area,
                    kwargs_cut=kwargs_source_cut,
                    variability_model=variability_model,
                    kwargs_variability_model=kwargs_variability,
                    list_type="astropy_table",
                    light_profile=source_light_profile,
                    catalog_type=catalog_type,
                )
            else:
                new_path = self.path + "/Sources/SupernovaeData/supernovae_data.pkl"
                with open(new_path, "rb") as f:
                    load_supernovae_data = pickle.load(f)
                self._sources = PointPlusExtendedSources(
                    load_supernovae_data,
                    cosmo=cosmo,
                    sky_area=sky_area,
                    kwargs_cut=kwargs_source_cut,
                    variability_model=variability_model,
                    kwargs_variability_model=kwargs_variability,
                    list_type="list",
                    light_profile=source_light_profile,
                )
            self._source_model_type = "point_plus_extended"
        else:
            raise ValueError("source_type %s is not supported" % source_type)
        self.cosmo = cosmo
        self.f_sky = sky_area

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
            )
            if gg_lens.validity_test(**kwargs_lens_cut):
                return gg_lens

    @property
    def deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that are being
        considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        return self._lens_galaxies.deflector_number()

    @property
    def source_number(self):
        """Number of sources that are being considered to be placed in the sky area
        potentially aligned behind deflectors.

        :return: number of potential sources
        """
        return self._sources.source_number_selected

    def get_num_sources_tested_mean(self, testarea):
        """Compute the mean of source galaxies needed to be tested within the test area.

        num_sources_tested_mean/ testarea = num_sources/ f_sky; testarea is in units of
        arcsec^2, f_sky is in units of deg^2. 1 deg^2 = 12960000 arcsec^2
        """
        num_sources = self._sources.source_number_selected
        num_sources_tested_mean = (testarea * num_sources) / (
            12960000 * self.f_sky.to_value("deg2")
        )
        return num_sources_tested_mean

    def get_num_sources_tested(self, testarea=None, num_sources_tested_mean=None):
        """Draw a realization of the expected distribution (Poisson) around the mean for
        the number of source galaxies tested."""
        if num_sources_tested_mean is None:
            num_sources_tested_mean = self.get_num_sources_tested_mean(testarea)
        num_sources_range = np.random.poisson(lam=num_sources_tested_mean)
        return num_sources_range

    def draw_population(self, kwargs_lens_cuts):
        """Return full population list of all lenses within the area # TODO: need to
        implement a version of it. (improve the algorithm)

        :param kwargs_lens_cuts: validity test keywords
        :type kwargs_lens_cuts: dict
        :return: List of Lens instances with parameters of the deflectors and lens and
            source light.
        :rtype: list
        """

        # Initialize an empty list to store the Lens instances
        gg_lens_population = []
        # Estimate the number of lensing systems
        num_lenses = self._lens_galaxies.deflector_number()
        # num_sources = self._source_galaxies.galaxies_number()
        #        print(num_sources_tested_mean)
        #        print("num_lenses is " + str(num_lenses))
        #        print("num_sources is " + str(num_sources))
        #        print(np.int(num_lenses * num_sources_tested_mean))

        # Draw a population of galaxy-galaxy lenses within the area.
        for _ in range(num_lenses):
            lens = self._lens_galaxies.draw_deflector()
            test_area = draw_test_area(deflector=lens)
            num_sources_tested = self.get_num_sources_tested(testarea=test_area)
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
                        source_type=self._source_model_type,
                        light_profile=self._sources.light_profile,
                        lightcurve_time=self.lightcurve_time,
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
