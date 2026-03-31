import warnings
from slsim.Sources.Supernovae import random_supernovae
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.ImageSimulation.image_quality_lenstronomy import get_all_supported_bands


class SupernovaEvent(SourceBase):
    """A class to manage a supernova."""

    def __init__(
        self,
        sn_type,
        sn_absolute_mag_band,
        sn_absolute_zpsys,
        lightcurve_time,
        variability_model,
        sn_modeldir=None,
        kwargs_variability=None,
        cosmo=None,
        **kwargs,
    ):
        """# TODO: is there a specific variability model needed for this class,
        if so, we should set it directly.

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
            :type sn_type: str
        :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
            :type variability_model: str
        :param kwargs_variability: Dictionary with bands as strings, each containing a dictionary for a
         Variability() class input configurations for point source variability
        :type kwargs_variability_model: dict of dict or None
        :param sn_absolute_mag_band: Band used to normalize to absolute magnitude
        :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
        :param sn_absolute_zpsys: Optional, AB or Vega (AB default)
        :type sn_absolute_zpsys: str
        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array
        :param sn_modeldir: sn_modeldir is the path to the directory containing files
            needed to initialize the sncosmo.model class. For example,
            sn_modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can
            be downloaded from https://github.com/LSST-strong-lensing/data_public .
            For more detail, please look at the documentation of RandomizedSupernovae
            class.
        :type sn_modeldir: str
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This table or dict should contain at least redshift of a supernova, offset from
         the host if host galaxy is available.
         eg: {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}
        :type source_dict: dict or astropy.table.Table
        :param cosmo: astropy.cosmology instance
        :param kwargs: dictionary of keyword arguments for a supernova for SourceBase() class.
         May be a dictionary or an Astropy table.
         This table or dict should contain atleast redshift of a supernova, offset from
         the host if host galaxy is available.
         eg: {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}
        """

        super().__init__(
            extended_source=False,
            point_source=True,
            cosmo=cosmo,
            variability_model=variability_model,
            **kwargs,
        )
        self.name = "SN" + sn_type
        self._variability_computed = False  # to be set to True once the light_curve() definition has been processed
        # These are the keywords that kwargs dict should contain
        self._kwargs_variability = kwargs_variability
        self._sn_type = sn_type
        self._sn_absolute_mag_band = sn_absolute_mag_band
        self._sn_absolute_zpsys = sn_absolute_zpsys
        self._lightcurve_time = lightcurve_time
        self._sn_modeldir = sn_modeldir

    @property
    def light_curve(self):
        """Provides lightcurves of a supernova in each band."""
        if self._kwargs_variability is not None:
            # Here we extract lightcurves of a supernova in given bands
            kwargs_variab_extracted = {}
            if self._cosmo is None:
                raise ValueError(
                    "Cosmology cannot be None for Supernova class. Please"
                    "provide a suitable astropy cosmology."
                )
            else:
                lightcurve_class = random_supernovae.RandomizedSupernova(
                    sn_type=self._sn_type,
                    redshift=self._z,
                    absolute_mag=None,
                    absolute_mag_band=self._sn_absolute_mag_band,
                    mag_zpsys=self._sn_absolute_zpsys,
                    cosmo=self._cosmo,
                    modeldir=self._sn_modeldir,
                )

            # Filter the input list against the global registry to ignore physical parameters
            supported_bands = get_all_supported_bands()
            provided_bands = set(supported_bands) & set(self._kwargs_variability)

            for element in provided_bands:
                # sncosmo registers LSST bands as 'lsstg', 'lsstr', etc.
                if element in ["u", "g", "r", "i", "z", "y"]:
                    provided_band = "lsst" + element
                else:
                    provided_band = element

                name = "ps_mag_" + element
                times = self._lightcurve_time

                # Safely attempt to generate the lightcurve
                try:
                    magnitudes = lightcurve_class.get_apparent_magnitude(
                        time=times,
                        band=provided_band,
                        zpsys=self._sn_absolute_zpsys,
                    )
                except Exception as e:
                    # If sncosmo throws an error, it means the band isn't registered
                    # in sncosmo's internal system. We skip it to avoid crashing.
                    warnings.warn(
                        f"Skipping band '{provided_band}': Failed to generate lightcurve. "
                        f"It may not be registered in sncosmo. (Error: {e})",
                        UserWarning,
                    )
                    continue

                # If successful, store the magnitudes
                if name not in self.source_dict:
                    self.source_dict[name] = float(min(magnitudes))

                kwargs_variab_extracted[element] = {
                    "MJD": times,
                    name: magnitudes,
                }
        else:
            kwargs_variab_extracted = {}

        self._variability_computed = True
        return kwargs_variab_extracted

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image. If None, takes the peak magnitude
        :type image_observation_times: array or None
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        # TODO: check whether image observation times are outside of light curve,
        #  then we can simply set the magnitude = -inf
        if not self._variability_computed:
            self._kwargs_variability_model = self.light_curve
        return super().point_source_magnitude(
            band=band, image_observation_times=image_observation_times
        )
