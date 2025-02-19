from slsim.Sources.source_pop_base import SourcePopBase
from slsim.Sources.galaxies import Galaxies
from slsim.selection import object_cut


class PointPlusExtendedSources(Galaxies, SourcePopBase):
    """Class to describe point and extended sources."""

    def __init__(
        self,
        point_plus_extended_sources_list,
        cosmo,
        sky_area,
        kwargs_cut,
        variability_model=None,
        kwargs_variability_model=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        light_profile="single_sersic",
        list_type="astropy_table",
        catalog_type=None,
        lightcurve_time=None,
        sn_type=None,
        sn_absolute_mag_band=None,
        sn_absolute_zpsys=None,
        sn_modeldir=None,
    ):
        """

        :param point_plus_extended_sources_list: list of dictionary with point and
         extended source parameters or astropy table of sources.
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         source.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual source.
        :param agn_driving_variability_model: Variability model with light_curve output
         which drives the variability across all bands of the agn. eg: "light_curve",
         "sinusoidal", "bending_power_law"
        :param agn_driving_kwargs_variability: Dictionary containing agn variability
         parameters for the driving variability class. eg: variable_agn_kwarg_dict =
         {"length_of_light_curve": 1000, "time_resolution": 1,
         "log_breakpoint_frequency": 1 / 20, "low_frequency_slope": 1,
         "high_frequency_slope": 3, "normal_magnitude_variance": 0.1}. For the detailed
          explanation of these parameters, see generate_signal() function in
          astro_util.py.
        :param light_profile: keyword for number of sersic profile to use in source
         light model. accepted kewords: "single_sersic", "double_sersic".
        :param list_type: format of the source catalog file. Currently, it supports
         a single astropy table or a list of astropy tables.
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it.
        :type catalog_type: str. eg: "scotch" or None
        :param lightcurve_time: Lightcurve observation time array in units of days. Defaults to None.
        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.). Defaults to None.
        :param sn_absolute_mag_band: Band used to normalize to absolute magnitude.
         Defaults to None.
        :param sn_absolute_zpsys: Zero point system, either AB or Vega, with None defaulting to AB.
         Defaults to None.
        :param sn_modeldir: sn_modeldir is the path to the directory containing files needed to initialize
         the sncosmo.model class. For example, sn_modeldir =
         'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can be downloaded
         from https://github.com/LSST-strong-lensing/data_public. For more detail,
         please look at the documentation of RandomizedSupernovae class. Defaults to None.
        """

        object_list = object_cut(
            point_plus_extended_sources_list,
            list_type=list_type,
            object_type="point",
            **kwargs_cut
        )
        Galaxies.__init__(
            self,
            galaxy_list=object_list,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut={},
            light_profile=light_profile,
            list_type=list_type,
            catalog_type=catalog_type,
        )
        SourcePopBase.__init__(
            self,
            cosmo=cosmo,
            sky_area=sky_area,
            variability_model=variability_model,
            kwargs_variability_model=kwargs_variability_model,
            agn_driving_variability_model=agn_driving_variability_model,
            agn_driving_kwargs_variability=agn_driving_kwargs_variability,
            lightcurve_time=lightcurve_time,
            sn_type=sn_type,
            sn_absolute_mag_band=sn_absolute_mag_band,
            sn_absolute_zpsys=sn_absolute_zpsys,
            sn_modeldir=sn_modeldir,
        )
        self.source_type = "point_plus_extended"
