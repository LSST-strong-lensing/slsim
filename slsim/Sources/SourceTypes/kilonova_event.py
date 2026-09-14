import warnings
import numpy as np
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Sources.Events.BNSMerger.kilonova import Kilonova
from slsim.ImageSimulation.image_quality_lenstronomy import (
    get_all_supported_bands,
    get_sncosmo_filtername,
)


class KilonovaEvent(SourceBase):
    """A class to manage a BNS merger."""

    def __init__(
        self,
        lightcurve_time,
        variability_model,
        model_name="mosfit_kilonova",
        mag_zpsys="AB",
        modeldir=None,
        kwargs_variability=None,
        kwargs_kilonova=None,
        cosmo=None,
        **kwargs,
    ):
        """
        :param lightcurve_time: Observation time array for the light curve in [days].
        :type lightcurve_time: array-like
        :param variability_model: Keyword for the variability model to be used. This is an
            input for the Variability class.
        :type variability_model: str
        :param model_name: Kilonova light curve model to be used. If not provided, the
            default model is the MOSFiT-based kilonova model.
        :type model_name: str
        :param mag_zpsys: Optional, AB or Vega (AB default).
        :type mag_zpsys: str
        :param modeldir: Directory including files for external kilonova models. This
            option is currently not supported.
        :type modeldir: str or None
        :param kwargs_variability: List containing the variability keyword and the
            bands for which the light curve should be generated.
        :type kwargs_variability: list or None
        :param kwargs_kilonova: Keyword arguments passed to the Kilonova class. This may
            include ejecta_mass in [solar masses], ejecta_velocity in [c], opacity in
            [cm^2 g^-1], temperature_floor in [K], and kappa_gamma in [cm^2 g^-1].
        :type kwargs_kilonova: dict or None
        :param cosmo: Astropy cosmology instance.
        :type cosmo: `~astropy.cosmology`
        :param kwargs: Keyword arguments passed to the SourceBase class. This may contain
            source properties such as redshift and offsets from the host galaxy in [arcsec].
        :type kwargs: dict
        """
        super().__init__(
            extended_source=False,
            point_source=True,
            cosmo=cosmo,
            variability_model=variability_model,
            **kwargs,
        )
        self.name = "BNS"
        self._variability_computed = False
        self._kwargs_variability = kwargs_variability
        self._lightcurve_time = lightcurve_time

        self._model_name = model_name
        self._mag_zpsys = mag_zpsys
        self._modeldir = modeldir

        if kwargs_kilonova is None:
            self._kwargs_kilonova = {}
        else:
            self._kwargs_kilonova = kwargs_kilonova

    @property
    def light_curve(self):
        """Provides lightcurves of a bns merger in each band."""
        if self._kwargs_variability is not None:
            kwargs_variab_extracted = {}
            if self._cosmo is None:
                raise ValueError(
                    "Cosmology cannot be None for BNSMerger class. Please"
                    "provide a suitable astropy cosmology."
                )
            else:
                # Initialize BNS/Kilonova light curve model
                lightcurve_class = Kilonova(
                    redshift=self._z,
                    model_name=self._model_name,
                    mag_zpsys=self._mag_zpsys,
                    cosmo=self._cosmo,
                    modeldir=self._modeldir,
                    **self._kwargs_kilonova,
                )
                self._lightcurve_class = lightcurve_class

            supported_bands = get_all_supported_bands()
            provided_bands = set(supported_bands) & set(self._kwargs_variability)

            for element in provided_bands:
                name = "ps_mag_" + element
                times = self._lightcurve_time

                # Convert SLSim short band labels, such as "i" and "r", to registered
                # filter names required by Redback, such as "lssti" and "lsstr".
                # We reuse the existing SLSim/SNcosmo filter-name helper for this mapping.
                provided_band = get_sncosmo_filtername(element)

                try:
                    magnitudes = lightcurve_class.get_apparent_magnitude(
                        time=times,
                        band=provided_band,
                        zpsys=self._mag_zpsys,
                    )
                    # make sure before and after the event, the flux is zero
                    magnitudes = np.append(np.inf, magnitudes)
                    magnitudes = np.append(magnitudes, np.inf)
                    padded_times = np.append(times[0] - (times[1] - times[0]), times)
                    padded_times = np.append(padded_times, 2 * times[-1] - times[-2])
                except Exception as e:
                    warnings.warn(
                        f"Skipping band '{provided_band}': Failed to generate lightcurve. "
                        f"(Error: {e})",
                        UserWarning,
                    )
                    continue

                if name not in self.source_dict:
                    self.source_dict[name] = float(min(magnitudes))

                kwargs_variab_extracted[element] = {
                    "MJD": padded_times,
                    name: magnitudes,
                }
        else:
            kwargs_variab_extracted = {}

        self._variability_computed = True
        return kwargs_variab_extracted

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the BNS/kilonova point source in a specific
        band.

        :param band: Imaging band.
        :type band: str
        :param image_observation_times: Image observation times in
            [days]. If None, takes the peak magnitude.
        :type image_observation_times: array-like or None
        :return: Magnitude of the point source in the specified band.
        :rtype: float or array-like
        """

        if not self._variability_computed:
            self._kwargs_variability_model = self.light_curve

        return super().point_source_magnitude(
            band=band, image_observation_times=image_observation_times
        )
