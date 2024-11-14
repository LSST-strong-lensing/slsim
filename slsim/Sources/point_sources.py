import numpy.random as random
from slsim.Sources.source_pop_base import SourcePopBase
import warnings
from slsim.selection import object_cut
from slsim.Sources.source import Source


class PointSources(SourcePopBase):
    """Class to describe point sources."""

    def __init__(
        self,
        point_source_list,
        cosmo,
        sky_area,
        kwargs_cut,
        variability_model=None,
        kwargs_variability_model=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        light_profile=None,
        list_type="astropy_table",
        lightcurve_time=None,
        sn_type=None,
        sn_absolute_mag_band=None,
        sn_absolute_zpsys=None,
        sn_modeldir=None
    ):
        """

        :param point_source_list: list of dictionary with quasar parameters or astropy
         table.
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max. These are
         the arguments that go into the deflector_cut() definition which is a general
         defination for performing given cuts in given catalog. For the supernovae
         sample, we can only apply redshift cuts because supernovae sample contains only
         redshift in this stage.
        :type kwargs_cut: dict
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         point source.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual point_source.
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
         light model. Always None for this class.
        :param list_type: type of the format of the source catalog. It should be either
         astropy_table or list of astropy table.
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

        self.n = len(point_source_list)
        self.light_profile = light_profile
        if self.light_profile is not None:
            warning_msg = (
                "The provided light profile %s is not used to describe the point "
                "source. The relevant light profile is None." % light_profile
            )
            warnings.warn(warning_msg, category=UserWarning, stacklevel=2)
        # make cuts
        self._point_source_select = object_cut(
            point_source_list, list_type=list_type, object_type="point", **kwargs_cut
        )

        self._num_select = len(self._point_source_select)
        super(PointSources, self).__init__(
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
            sn_modeldir=sn_modeldir
            )
        self.source_type = "point_source"

    @property
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        number = self.n
        return number

    @property
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        return self._num_select

    def draw_source(self):
        """Choose source at random with the selected range.

        :return: dictionary of source
        """

        index = random.randint(0, self._num_select - 1)
        point_source = self._point_source_select[index]
        source_class = Source(
                    source_dict=point_source,
                    variability_model=self.variability_model,
                    kwargs_variability=self.kwargs_variability,
                    sn_type=self.sn_type,
                    sn_absolute_mag_band=self.sn_absolute_mag_band,
                    sn_absolute_zpsys=self.sn_absolute_zpsys,
                    cosmo=self._cosmo,
                    lightcurve_time=self.lightcurve_time,
                    sn_modeldir=self.sn_modeldir,
                    agn_driving_variability_model=self.agn_driving_variability_model,
                    agn_driving_kwargs_variability=self.agn_driving_kwargs_variability,
                    source_type=self.source_type,
                    light_profile=self.light_profile,
                )

        return source_class
