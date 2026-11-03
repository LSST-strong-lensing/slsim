import numpy as np
import json
from pathlib import Path
import mosfit
import speclite.filters
from slsim.Pipelines.roman_speclite import configure_roman_filters
from slsim.ImageSimulation.image_quality_lenstronomy import get_sncosmo_filtername


class Kilonova:
    """Generate kilonova light curves using Redback or MOSFiT."""

    def __init__(
        self,
        redshift,
        model_name="mosfit_kilonova",
        model_parameters=None,
        mag_zpsys="AB",
        cosmo=None,
        backend="redback",
        **kwargs,
    ):
        """
        :param redshift: The redshift of the kilonova source.
        :type redshift: float
        :param model_name: The kilonova light curve model to be used. If not provided,
            the default model is the MOSFiT-based kilonova model.
        :type model_name: str
        :param model_parameters: Physical parameters passed to the selected model.
            Dictionary keys are parameter names and values are parameter values.
        :type model_parameters: dict or None
        :param mag_zpsys: Optional, AB or Vega (AB default).
        :type mag_zpsys: str
        :param cosmo: Cosmology for luminosity distance calculation.
        :type cosmo: `~astropy.cosmology`
        :param backend: Backend used to calculate the light curve, choosing from
            "redback" or "mosfit". Defaults to "redback".
        :type backend: str
        :param kwargs: Additional keyword arguments passed to the Redback kilonova model.
        :type kwargs: dict
        """

        if backend not in ("redback", "mosfit"):
            raise ValueError("backend must be 'redback' or 'mosfit'.")

        if not np.isfinite(redshift) or redshift <= 0:
            raise ValueError("redshift must be finite and positive.")

        if model_parameters is None:
            model_parameters = {}

        if not isinstance(model_parameters, dict):
            raise TypeError("model_parameters must be a dictionary.")

        overlap = model_parameters.keys() & kwargs.keys()
        if overlap:
            raise ValueError(
                f"Parameters supplied twice: {sorted(overlap)}. "
                "Provide each parameter in either model_parameters or kwargs."
            )

        self._model_parameters = model_parameters.copy()
        self._model_parameters.update(kwargs)

        self._backend = backend
        self._model_name = model_name
        self._redshift = redshift
        self._mag_zpsys = "AB"
        self._cosmo = cosmo

        if self._backend == "redback":
            from redback.transient_models import kilonova_models

            self._model = getattr(kilonova_models, self._model_name)

        elif self._backend == "mosfit":
            from mosfit.fitter import Fitter

            self._fitter = Fitter(quiet=True, exit_on_prompt=True)

    def get_apparent_magnitude(self, time, band):
        """Function to return apparent magnitude of a kilonova for a given band
        and time with Redback model.

        :param time: The observer-frame time array to evaluate the model
            (in days)
        :type time: array-like
        :param band: The band to evaluate the model over.
        :type band: str or list
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str
        :return: magnitude of source
        """
        if self._backend == "redback":
            provided_band = get_sncosmo_filtername(band)

            return self._model(
                time=time,
                redshift=self._redshift,
                bands=provided_band,
                output_format="magnitude",
                cosmology=self._cosmo,
                **self._model_parameters,
            )

        elif self._backend == "mosfit":
            return self._get_mosfit_magnitude(time, band)

    def _get_roman_filter(self, band):
        configure_roman_filters()

        return speclite.filters.load_filter("Roman-" + band)

    def _configure_mosfit_filter(self, band):
        roman_filter = self._get_roman_filter(band)
        filter_name = "SLSim_Roman_" + band

        directory = Path.cwd() / "modules" / "observables"
        directory.mkdir(parents=True, exist_ok=True)

        # save curve files
        filter_path = directory / (filter_name + ".dat")
        np.savetxt(
            filter_path,
            np.column_stack(
                [
                    roman_filter.wavelength,
                    roman_filter.response,
                ]
            ),
        )

        # read path of filter setup
        rules_path = directory / "filterrules.json"
        source_path = (
            rules_path
            if rules_path.exists()
            else (
                Path(mosfit.__file__).resolve().parent
                / "modules"
                / "observables"
                / "filterrules.json"
            )
        )

        with open(source_path) as file:
            rules = json.load(file)

        for rule in rules:
            rule.get("filters", {}).pop(filter_name, None)

        rules.append(
            {
                "systems": ["AB"],
                "instruments": ["SLSimRoman"],
                "filters": {
                    filter_name: {
                        "path": str(filter_path.resolve()),
                        "xunit": "Angstrom",
                        "offset": 0.0,
                    }
                },
            }
        )

        with open(rules_path, "w") as file:
            json.dump(rules, file, indent=2)

        return filter_name

    def _get_mosfit_magnitude(self, time, band):
        filter_name = self._configure_mosfit_filter(band)

        parameters = self._model_parameters.copy()
        parameters.update(
            redshift=self._redshift,
            lumdist=self._cosmo.luminosity_distance(self._redshift).to_value("Mpc"),
            texplosion=0.0,
        )

        fixed_parameters = []
        for name, value in parameters.items():
            fixed_parameters.extend([name, str(value)])

        results = self._fitter.fit_events(
            models=[self._model_name],
            time_list=list(time),
            max_time=float(np.max(time)),
            band_list=[filter_name],
            band_systems=["AB"],
            band_instruments=["SLSimRoman"],
            user_fixed_parameters=fixed_parameters,
            iterations=0,
            num_walkers=1,
            write=False,
            return_fits=True,
            exit_on_prompt=True,
        )

        # Select the model magnitudes for this band from the complete results
        photometry = [
            row
            for row in results[0][0][0]["photometry"]
            if "realization" in row
            and "magnitude" in row
            and row.get("band") == filter_name
        ]

        # magnitude for input time
        magnitudes = []
        for t in time:
            values = [
                float(row["magnitude"])
                for row in photometry
                if np.isclose(float(row["time"]), t, rtol=1e-9, atol=1e-7)
            ]
            if not values:
                raise RuntimeError(f"No magnitude returned at {t} days.")
            magnitudes.append(values[0])

        return np.asarray(magnitudes)
