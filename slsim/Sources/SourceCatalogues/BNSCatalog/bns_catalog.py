import numpy as np
from astropy import units
from astropy.table import Table
from slsim.Sources.Events.event_lightcone import EventLightcone
from slsim.Sources.Events.BNSMerger.bns_mass_distribution import BNSMassDistribution


class BNSCatalog:
    """Class to generate a catalog of binary neutron star (BNS) merger events
    within a selected sky area."""

    def __init__(
        self,
        cosmo,
        sky_area,
        band_list,
        lightcurve_time,
        kwargs_kilonova=None,
        z_max=5,
        time_interval=1 * units.year,
        noise=True,
        model_name="mosfit_kilonova",
        mag_zpsys="AB",
        modeldir=None,
        backend="redback",
    ):
        """
        :param cosmo: Astropy cosmology instance.
        :type cosmo: `~astropy.cosmology`
        :param sky_area: sky area for sampled event in [solid angle]
        :type sky_area: `~Astropy.units.Quantity`
        :param band_list: Imaging bands for which the kilonova light curves
            are generated.
        :type band_list: list or str
        :type band_list: list or str
        :param lightcurve_time: Observation time array for the light curve in [days]
        :type lightcurve_time: array-like
        :param kwargs_kilonova: Keyword arguments for the default Redback
            ``redback.transient_models.kilonova_models.mosfit_kilonova`` model,
            passed through the Kilonova class.
            If None, defaults are used for the Redback mosfit_kilonova model.
            The default opacities of 0.5, 3.0, and 10.0 [cm^2 g^-1] follow the
            three-component kilonova model of Villar et al. (2017). The remaining
            default values for the ejecta masses [solar masses], ejecta velocities
            [c], temperature floors [K], and gamma-ray opacity [cm^2 g^-1] are
            representative choices for generating the default light curve.
            User-provided values override the corresponding defaults.
            Other models receive only the supplied parameters. For MOSFiT bns,
            Mchirp and q are assigned from each event\'s sampled masses.
        :type kwargs_kilonova: dict or None
        :param z_max: Maximum redshift up to which BNS events are sampled
        :type z_max: float
        :param time_interval: time interval for event density lightcone to be evaluated over
        :type time_interval: `~Astropy.units.Quantity`
        :param noise: poisson-sample the number of event in the event density lightcone
        :type noise: bool
        :param model_name: The kilonova light curve model to be used. The
            model name must match a model implemented in the selected backend. The parameters in
            ``kwargs_kilonova`` must match those required by the selected model.
            If not provided, the default model is ``mosfit_kilonova``.
        :type model_name: str
        :param mag_zpsys: Optional, AB or Vega (AB default)
        :type mag_zpsys: str or None
        :param modeldir: Retained for compatibility; external models are unsupported.
        :type modeldir: str or None
        :param backend: Light curve backend, "redback" or "mosfit".
        :type backend: str
        """

        self._cosmo = cosmo
        self._sky_area = sky_area
        self._band_list = band_list
        self._lightcurve_time = lightcurve_time
        self._z_max = z_max
        self._time_interval = time_interval
        self._noise = noise
        self._model_name = model_name
        self._mag_zpsys = mag_zpsys
        self._modeldir = modeldir
        self._backend = backend

        default_kwargs_kilonova = {
            "mej_1": 0.01,
            "mej_2": 0.02,
            "mej_3": 0.03,
            "vej_1": 0.1,
            "vej_2": 0.2,
            "vej_3": 0.3,
            "kappa_1": 0.5,
            "kappa_2": 3.0,
            "kappa_3": 10.0,
            "temperature_floor_1": 5000,
            "temperature_floor_2": 4000,
            "temperature_floor_3": 3000,
            "kappa_gamma": 10,
        }

        if backend == "redback" and model_name == "mosfit_kilonova":
            self._kwargs_kilonova = {
                **default_kwargs_kilonova,
                **(kwargs_kilonova or {}),
            }
        else:
            self._kwargs_kilonova = (
                {} if kwargs_kilonova is None else kwargs_kilonova.copy()
            )

    def bns_catalog(self):
        redshifts = np.linspace(0, self._z_max, 500)

        bns_lightcone = EventLightcone(
            cosmo=self._cosmo,
            redshifts=redshifts,
            sky_area=self._sky_area,
            noise=self._noise,
            time_interval=self._time_interval,
            model="BNS",
        )
        bns_redshifts = bns_lightcone.event_sample()
        bns_table = Table({"z": bns_redshifts})
        bns_table = BNSMassDistribution().assign_masses(bns_table)
        bns_table["lightcurve_time"] = np.tile(
            self._lightcurve_time,
            (len(bns_table), 1),
        )

        bns_table["model_name"] = [self._model_name] * len(bns_table)
        bns_table["variability_model"] = ["light_curve"] * len(bns_table)
        bns_table["mag_zpsys"] = [self._mag_zpsys] * len(bns_table)

        bns_table["backend"] = [self._backend] * len(bns_table)

        event_parameters = np.empty(len(bns_table), dtype=object)

        for index, event in enumerate(bns_table):
            parameters = self._kwargs_kilonova.copy()

            if self._backend == "mosfit" and self._model_name == "bns":
                parameters["Mchirp"] = float(event["Mchirp"])
                parameters["q"] = float(event["q"])

            event_parameters[index] = parameters

        bns_table["kwargs_kilonova"] = event_parameters

        return bns_table
