from astropy import cosmology
from redback.transient_models import kilonova_models


class Kilonova:
    """Class for initializing a kilonova light curve model.

    If modeldir is provided, external kilonova model files are expected.
    This option is currently not supported. If modeldir is not provided,
    the model is retrieved from Redback's built-in kilonova models. By
    default, the MOSFiT-based kilonova model is used. Information about
    Redback can be found at
    https://redback.readthedocs.io/en/latest/.

    Following the GW170817 three-component convention, components 1, 2,
    and 3 can represent the blue, purple, and red ejecta components when
    assigned low, intermediate, and high opacities, respectively. This
    correspondence is not enforced, and users may specify the parameters
    of each component independently.
    """

    def __init__(
        self,
        redshift,
        mej_1,
        mej_2,
        mej_3,
        vej_1,
        vej_2,
        vej_3,
        kappa_1,
        kappa_2,
        kappa_3,
        temperature_floor_1,
        temperature_floor_2,
        temperature_floor_3,
        model_name="mosfit_kilonova",
        kappa_gamma=10,
        mag_zpsys="AB",
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        modeldir=None,
        **kwargs,
    ):
        """
        :param redshift: The redshift of the kilonova source.
        :type redshift: float

        :param mej_1: Ejecta mass of model component 1 in [M_sun], which
            sets the amount of radiating material and affecting the photon diffusion
            timescale.
        :type mej_1: float
        :param mej_2: Ejecta mass of model component 2, with the same
            definition and units as ``mej_1``.
        :type mej_2: float
        :param mej_3: Ejecta mass of model component 3, with the same
            definition and units as ``mej_1``.
        :type mej_3: float
        :param vej_1: Expansion velocity of model component 1 in units of the
            speed of light [c], affecting its expansion and photon diffusion
            timescale.
        :type vej_1: float
        :param vej_2: Expansion velocity of model component 2, with the same
            definition and units as ``vej_1``.
        :type vej_2: float
        :param vej_3: Expansion velocity of model component 3, with the same
            definition and units as ``vej_1``.
        :type vej_3: float
        :param kappa_1: Effective gray opacity of model component 1 in
            [cm^2 g^-1], controlling how readily radiation escapes from the
            ejecta.
        :type kappa_1: float
        :param kappa_2: Effective gray opacity of model component 2, with the
            same definition and units as ``kappa_1``.
        :type kappa_2: float
        :param kappa_3: Effective gray opacity of model component 3, with the
            same definition and units as ``kappa_1``.
        :type kappa_3: float
        :param temperature_floor_1: Minimum effective photospheric temperature of model
            component 1 in [K], affecting its late-time spectral evolution.
        :type temperature_floor_1: float
        :param temperature_floor_2: Minimum effective photospheric temperature of model
            component 2, with the same definition and units as ``temperature_floor_1``.
        :type temperature_floor_2: float
        :param temperature_floor_3: Minimum effective photospheric temperature of model
            component 3, with the same definition and units as ``temperature_floor_1``.
        :type temperature_floor_3: float

        :param model_name: The kilonova light curve model to be used. If not provided,
            the default model is the MOSFiT-based kilonova model.
        :type model_name: str
        :param kappa_gamma: Gamma-ray opacity shared by all three model components
            in [cm^2 g^-1], controlling the trapping and escape of high-energy radiation
            from radioactive decay.
        :type kappa_gamma: float
        :param mag_zpsys: Optional, AB or Vega (AB default).
        :type mag_zpsys: str
        :param cosmo: Cosmology for luminosity distance calculation.
        :type cosmo: `~astropy.cosmology`
        :param modeldir: Directory including files for external kilonova models.
        :type modeldir: str or None
        :param kwargs: Additional keyword arguments passed to the Redback kilonova model.
        :type kwargs: dict
        """

        if modeldir is not None:
            raise NotImplementedError(
                "Only built-in Redback kilonova models are currently supported. "
                "External kilonova model files are not supported."
            )

        if not hasattr(kilonova_models, model_name):
            raise ValueError(
                f"Unsupported kilonova model '{model_name}'. "
                "The model must be available in "
                "redback.transient_models.kilonova_models."
            )

        self._model = getattr(kilonova_models, model_name)

        self._model_name = model_name
        self._redshift = redshift
        self._mag_zpsys = mag_zpsys
        self._cosmo = cosmo
        self._kwargs = kwargs

        self._model_parameters = {
            "mej_1": mej_1,
            "mej_2": mej_2,
            "mej_3": mej_3,
            "vej_1": vej_1,
            "vej_2": vej_2,
            "vej_3": vej_3,
            "kappa_1": kappa_1,
            "kappa_2": kappa_2,
            "kappa_3": kappa_3,
            "temperature_floor_1": temperature_floor_1,
            "temperature_floor_2": temperature_floor_2,
            "temperature_floor_3": temperature_floor_3,
            "kappa_gamma": kappa_gamma,
        }

    def get_apparent_magnitude(self, time, band, zpsys="AB"):
        """Function to return apparent magnitude of a kilonova for a given band
        and time.

        :param time: The observer-frame time array to evaluate the model
            (in days)
        :type time: array-like
        :param band: The band to evaluate the model over.
        :type band: str or list
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str
        :return: magnitude of source
        """

        return self._model(
            time=time,
            redshift=self._redshift,
            bands=band,
            output_format="magnitude",
            cosmology=self._cosmo,
            **self._model_parameters,
            **self._kwargs,
        )
