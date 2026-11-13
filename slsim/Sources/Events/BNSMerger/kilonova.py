import numpy as np
import speclite.filters
from astropy.table import Table
from astropy import units as u
from slsim.ImageSimulation.image_quality_lenstronomy import get_speclite_filtername
from redback.transient_models import kilonova_models


class Kilonova:
    """Generate kilonova light curves using Redback."""

    def __init__(
        self,
        redshift,
        model_name="mosfit_kilonova",
        model_parameters=None,
        mag_zpsys="AB",
        cosmo=None,
        modeldir=None,
        **kwargs,
    ):
        """
        :param redshift: The redshift of the kilonova source.
        :type redshift: float
        :param model_name: Name of a Redback kilonova model, such as ``mosfit_kilonova``
            or ``nicholl_bns``.
        :type model_name: str
        :param model_parameters: Physical parameters passed to the selected model.
            Dictionary keys are parameter names and values are parameter values.
        :type model_parameters: dict or None
        :param mag_zpsys: Optional, AB or Vega (AB default).
        :type mag_zpsys: str
        :param cosmo: Cosmology for luminosity distance calculation.
        :type cosmo: `~astropy.cosmology`
        :param kwargs: Additional keyword arguments passed to the Redback kilonova model.
        :type kwargs: dict
        """

        if modeldir is not None:
            raise NotImplementedError(
                "Only built-in Redback kilonova models are currently supported. "
                "External kilonova model files are not supported."
            )

        if model_name not in ("mosfit_kilonova", "nicholl_bns"):
            raise ValueError(
                f"Unsupported kilonova model '{model_name}'. "
                "Please choose 'mosfit_kilonova' or 'nicholl_bns'."
            )

        if not np.isfinite(redshift) or redshift <= 0:
            raise ValueError("redshift must be finite and positive.")

        if model_parameters is None:
            model_parameters = {}

        if not isinstance(model_parameters, dict):
            raise TypeError("model_parameters must be a dictionary.")

        self._model_parameters = model_parameters.copy()
        self._model_parameters.update(kwargs)

        self._model_name = model_name
        self._redshift = redshift
        self._mag_zpsys = "AB"
        self._cosmo = cosmo
        self._model = getattr(kilonova_models, self._model_name)

    def get_apparent_magnitude(self, time, band):
        """Function to return apparent magnitude of a kilonova for a given band
        and time with Redback models.

        :param time: The observer-frame time array to evaluate the model
            (in days)
        :type time: array-like
        :param band: The band to evaluate the model over.
        :type band: str or list
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str
        :return: magnitude of source
        """

        spectrum = self._model(
            time=time,
            redshift=self._redshift,
            cosmology=self._cosmo,
            output_format="spectra",
            **self._model_parameters,
        )

        wavelength = np.asarray(spectrum.lambdas) * u.AA
        filter_name = get_speclite_filtername(band)
        current_filter = speclite.filters.load_filter(filter_name)
        magnitudes = current_filter.get_ab_magnitude(spectrum.spectra, wavelength)

        return Table(
            {
                "time": spectrum.time,
                "magnitude": magnitudes,
            }
        )
