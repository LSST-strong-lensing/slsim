import os
import numpy as np
import sncosmo
from astropy import cosmology
from slsim.Sources.Supernovae.supernova import Supernova

_ABSOLUTE_MAG_DISTS = {
    "Ia": [-19.37, 0.47],
    "Ib": [-17.90, 0.90],
    "Ic": [-18.30, 0.60],
    "IIb": [-17.03, 0.93],
    "IIL": [-17.98, 0.90],
    "IIP": [-16.80, 0.97],
    "IIn": [-18.62, 1.48],
    "Ic-BL": [-18.30, 0.60],
}


def get_accepted_sn_types():
    """Helper function to get SN types from the SNCosmo source classes.
    The accepted sn_types are:
    [II, II-pec, IIL, IIL/P, IIP, IIb, IIn, Ia, Ib, Ib/c, Ic, Ic-BL, PopIII]

    :return: dictionary of types and sources, and list of models
    """

    all_models = sncosmo.registry._get_registry(sncosmo.Source)
    all_models_type_dict = {
        mod[0]: all_models._loaders[mod][2]["type"].split()[-1]
        for mod in all_models._loaders.keys()
    }
    return all_models_type_dict, list(np.unique(list(all_models_type_dict.values())))


class RandomizedSupernova(Supernova):
    """Class for randomizing a supernova of the type sn_type specified by the
    user.

    If modeldir is not provided by the user, and sn_type is Ia, the sn_model is
    chosen to be salt3-nir.
    If modeldir is not provided by the user and sn_type is other than Ia, the
    sn_model is picked randomly from a list of built-in models with the same sn_type

    If modeldir is provided by the user and sn_type is Ia, the sn_model is
    chosen to be salt3. In this case, modeldir is the path to the directory
    containing files needed to initialize the sncosmo.SALT3Source class.
    For example, modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'

    If modeldir is provided by the user and sn_type is other than Ia, the
    sn_model is picked randomly from the list of models located in modeldir.
    In this case, modeldir is the path to the full list of models. For example,
    modeldir = 'C:/Users/username/Documents/NON1ASED.V19_CC+HostXT_WAVEEXT
    If the specified sn_type is Ic, then a supernova of type Ic will be picked
    at random.

    These files can be found in https://github.com/LSST-strong-lensing/data_public
    """

    def __init__(
        self,
        sn_type,
        redshift,
        absolute_mag=None,
        absolute_mag_band="bessellb",
        mag_zpsys="AB",
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        modeldir=None,
        random_seed=None,
        **kwargs
    ):
        """

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param redshift: The redshift of the source.
        :type redshift: float
        :param absolute_mag: Absolute magnitude of the supernova
        :type absolute_mag: float
        :param absolute_mag_band: Band used to normalize to absolute magnitude
        :type absolute_mag_band: str or `~sncosmo.Bandpass`
        :param mag_zpsys: Optional, AB or Vega (AB default)
        :type mag_zpsys: str
        :param cosmo: Cosmology for absolute magnitude
        :type cosmo: `~astropy.cosmology`
        :param modeldir: Path to the directory containing supernova files
        :type modeldir: str
        :param random_seed: Random seed for randomization
        :type random_seed: int or None
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        all_models, accepted_types = get_accepted_sn_types()
        if sn_type not in accepted_types:
            raise RuntimeError(
                "You passed %s as your SN type, " % sn_type
                + "but currently accepted SN types are: "
                + ", ".join(accepted_types)
            )
        self._sn_type = sn_type
        self._absolute_mag_band = absolute_mag_band

        if absolute_mag is None:
            absolute_mag = self.get_absolute_magnitude(self._sn_type)

        if modeldir is None:
            self._accepted_SN_types = accepted_types
            self._all_sncosmo_models = all_models
            self._type_models = None
            self.set_random_sed_model(self._sn_type)
        elif sn_type == "Ia":
            self._sncosmo_source = "salt3"
        else:
            source_list = [
                source for source in os.listdir(os.path.join(modeldir, sn_type))
            ]
            random_source = source_list[np.random.randint(0, len(source_list))]
            self._sncosmo_source = str(random_source)[:-4]

        Supernova.__init__(
            self,
            source=self._sncosmo_source,
            redshift=redshift,
            sn_type=self._sn_type,
            absolute_mag=absolute_mag,
            absolute_mag_band=absolute_mag_band,
            mag_zpsys=mag_zpsys,
            cosmo=cosmo,
            modeldir=modeldir,
            **kwargs
        )

        if self._sn_type == "Ia":
            self.set(**{"c": np.random.normal(0, 0.1), "x1": np.random.normal(0, 1)})

    def set_random_sed_model(self, sn_type):
        """Function to set a random SED model for a given SN type.

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param random_seed: Random seed for randomization
        :type random_seed: int

        :return: randomized `~sncosmo.Source` class
        """
        if sn_type not in self._accepted_SN_types:
            raise RuntimeError(
                "You passed %s as your SN type, " % sn_type
                + "but currently accepted SN types are: "
                + ", ".join(self._accepted_SN_types)
            )

        if sn_type == "Ia":
            self._sncosmo_source = "salt3-nir"
            return

        if self._type_models is None:
            if sn_type == "IIP":
                check_types = ["IIP", "II"]
            else:
                check_types = [sn_type]

            type_models = [
                mod
                for mod in self._all_sncosmo_models.keys()
                if np.any([typ in self._all_sncosmo_models[mod] for typ in check_types])
            ]
            self._type_models = type_models

        random_ind = np.random.randint(0, len(self._type_models))

        self._sncosmo_source = self._type_models[random_ind]
        return self._sncosmo_source

    def get_absolute_magnitude(self, sn_type, absolute_mag_distribution=None):
        """Function to get a reasonable absolute mag for a given SN type.

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param absolute_mag_distribution: A function that returns an
            absolute mag
        :type absolute_mag_distribution: func
        :param random_seed: Random seed for randomization
        :type random_seed: int
        :return: absolute magnitude of the source in the B band
        """

        if absolute_mag_distribution is None:
            mu, sigma = _ABSOLUTE_MAG_DISTS[sn_type]
            absolute_mag = np.random.normal(mu, sigma)
            self._absolute_mag_band = "bessellb"
        else:
            absolute_mag = absolute_mag_distribution()
        return absolute_mag
