import warnings
import numpy as np
from slsim.Sources.Events.Supernovae import random_supernovae
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.ImageSimulation.image_quality_lenstronomy import (
    get_all_supported_bands,
    get_sncosmo_filtername,
)


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
        name=None,
        **kwargs,
    ):
        """# TODO: is there a specific variability model needed for this class,
        if so, we should set it directly.

        Parameters fall into two categories:

        - **Required per-object catalog data** (passed via ``kwargs``/``source_dict``,
          typically one row of a catalog Table): at minimum ``z`` (redshift), plus
          ``ra_off``/``dec_off`` if a host galaxy is present. See ``kwargs`` below.
        - **Population-level config** (the explicit named arguments below --
          ``sn_type``, ``sn_absolute_mag_band``, ``sn_absolute_zpsys``,
          ``lightcurve_time``, ``variability_model``, ``sn_modeldir``,
          ``kwargs_variability``): normally the same value for every draw, but any
          of these may be overridden on a per-object basis by including a
          same-named column in the input catalog (see `PointSources`/
          `PointPlusExtendedSources`, which merge population-level config with the
          per-object catalog row, catalog value taking precedence).

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
        :type variability_model: str
        :param kwargs_variability: Dictionary with bands as strings, each containing a dictionary for a
         Variability() class input configurations for point source variability
        :type kwargs_variability: dict of dict or None
        :param sn_absolute_mag_band: Band used to normalize to absolute magnitude
        :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
        :param sn_absolute_zpsys: Optional, AB or Vega (AB default)
        :type sn_absolute_zpsys: str
        :param lightcurve_time: time array for lightcurve in unit of days,
         defined in the rest (source) frame relative to time_zero_point. This matches the
         convention used by Quasar() and is what Source._image_to_source_time_translation()
         assumes when converting observer-frame query times before a light curve lookup.
         Internally this gets converted to observer-frame time (multiplied by (1+z)) only
         for the call into sncosmo, which expects observer-frame time and applies its own
         (1+z) dilation.
        :type lightcurve_time: array
        :param sn_modeldir: sn_modeldir is the path to the directory containing files
            needed to initialize the sncosmo.model class. For example,
            sn_modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can
            be downloaded from https://github.com/LSST-strong-lensing/data_public .
            For more detail, please look at the documentation of RandomizedSupernovae
            class.
        :type sn_modeldir: str
        :param cosmo: astropy.cosmology instance
        :param kwargs: required per-object catalog data (dict or one row of an
         Astropy table), passed through to `SourceBase`. Must contain at least
         redshift (``z``); include ``ra_off``/``dec_off`` (offset from host galaxy
         center, in arcsec) if a host galaxy is available.
         eg: {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}
        """
        if name is None:
            name = "SN" + sn_type
        super().__init__(
            extended_source=False,
            point_source=True,
            cosmo=cosmo,
            variability_model=variability_model,
            name=name,
            **kwargs,
        )

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
            lightcurve_class = random_supernovae.RandomizedSupernova(
                sn_type=self._sn_type,
                redshift=self._z,
                absolute_mag=None,
                absolute_mag_band=self._sn_absolute_mag_band,
                mag_zpsys=self._sn_absolute_zpsys,
                cosmo=self._cosmo,
                modeldir=self._sn_modeldir,
            )
            self._lightcurve_class = lightcurve_class

            # Filter the input list against the global registry to ignore non-band parameters and unrecognized bands
            supported_bands = get_all_supported_bands()
            provided_bands = set(supported_bands) & set(self._kwargs_variability)

            # sncosmo expects observer-frame times, so we convert to observer-frame times using the redshift
            times = self._lightcurve_time
            observer_frame_times = times * (1 + self._z)

            for element in provided_bands:
                # sncosmo registers LSST bands as 'lsstg', 'lsstr', etc.
                provided_band = get_sncosmo_filtername(element)

                name = "ps_mag_" + element

                # Safely attempt to generate the lightcurve
                try:
                    magnitudes = lightcurve_class.get_apparent_magnitude(
                        time=observer_frame_times,
                        band=provided_band,
                        zpsys=self._sn_absolute_zpsys,
                    )
                    # make sure before and after the event, the flux is zero
                    magnitudes = np.append(np.inf, magnitudes)
                    magnitudes = np.append(magnitudes, np.inf)
                    padded_times = np.append(times[0] - (times[1] - times[0]), times)
                    padded_times = np.append(padded_times, 2 * times[-1] - times[-2])
                except Exception as e:
                    # If sncosmo throws an error, it means the band isn't registered
                    # in sncosmo's internal system. We skip it to avoid crashing.
                    warnings.warn(
                        f"Skipping band '{provided_band}': Failed to generate lightcurve. "
                        f"It may not be registered in sncosmo. (Error: {e})",
                        UserWarning,
                    )
                    continue

                # If successful, store the magnitudes. The light curve is infinite
                # wherever the supernova has no flux, so a peak magnitude only
                # exists if something finite is left.
                peak_magnitude = np.nanmin(magnitudes)
                if name not in self.source_dict and np.isfinite(peak_magnitude):
                    self.source_dict[name] = float(peak_magnitude)

                kwargs_variab_extracted[element] = {
                    "MJD": padded_times,
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

    def update_microlensing_kwargs_source_morphology(self, kwargs_source_morphology):
        """Injects the sncosmo model instance into morphology kwargs so the
        morphology uses the exact same SN realisation — template, x1, c — as
        the lightcurve."""
        if not self._variability_computed:
            _ = self.light_curve  # ensures _lightcurve_class is populated

        if hasattr(self, "_lightcurve_class"):
            kwargs_source_morphology.setdefault(
                "sn_model_instance", self._lightcurve_class
            )
        return kwargs_source_morphology
