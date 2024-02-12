import numpy as np
import sncosmo
from astropy import cosmology
from slsim.Util import param_util

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
    all_models = sncosmo.registry._get_registry(sncosmo.Source)
    all_models_type_dict = {
        mod[0]: all_models._loaders[mod][2]["type"].split()[-1]
        for mod in all_models._loaders.keys()
    }
    return all_models_type_dict, list(np.unique(list(all_models_type_dict.values())))


class Supernova(sncosmo.Model):
    """Class describing a supernova."""

    def __init__(
        self,
        source,
        redshift,
        sn_type,
        absolute_mag=None,
        absolute_mag_band=None,
        peak_apparent_mag=None,
        peak_apparent_mag_band=None,
        mag_zpsys="AB",
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        **kwargs
    ):
        """

        :param source: The model for the spectral evolution of the source. If a string
            is given, it is used to retrieve a `~sncosmo.Source` from
            the registry.
        :type source: `~sncosmo.Source` or str
        :param redshift: The redshift of the source.
        :type redshift: float
        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param absolute_mag: Absolute magnitude of the supernova
        :type absolute_mag: float
        :param absolute_mag_band: Band used to normalize to absolute magnitude
        :type absolute_mag_band: str or `~sncosmo.Bandpass`
        :param peak_apparent_mag: Peak apparent mag of the supernova
        :type peak_apparent_mag: str or `~sncosmo.Bandpass`
        :param peak_apparent_mag_band: Band used to normalize to apparent magnitude
        :type peak_apparent_mag_band: str or `~sncosmo.Bandpass`
        :param mag_zpsys: Optional, AB or Vega (AB default)
        :type mag_zpsys: str
        :param cosmo: Cosmology for absolute magnitude
        :type cosmo: `~astropy.cosmology`
        """
        super(Supernova, self).__init__(source=source, **kwargs)

        self._parameters[0] = redshift
        self._sn_type = sn_type
        if absolute_mag is not None:
            if absolute_mag_band is None:
                print(
                    "Must set absolute_mag_band when attempting to set an absolute magnitude."
                )
            else:
                self.set_source_peakabsmag(
                    absolute_mag, absolute_mag_band, mag_zpsys, cosmo=cosmo
                )
                if peak_apparent_mag is not None:
                    print(
                        "Both peak_apparent_mag and absolute_mag were given, choosing absolute_mag."
                    )
        elif peak_apparent_mag is not None:
            self.set_source_peakmag(
                peak_apparent_mag, peak_apparent_mag_band, mag_zpsys
            )
        else:
            print(
                "Warning, you should use self.set_source_peakabsmag or sefl.set_source_peakmag to set the amplitude."
            )

    def get_apparent_magnitude(self, time, band, zpsys="AB"):
        """Function to return apparent magnitude of a SN for a given band and time.

        :param time: The observer-frame time array to evaluate the model
        :type time: `~np.ndarray` or list
        :param band: The bandpass to evaluate the model over
        :type band: str or `~sncosmo.Bandpass`
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str
        """

        return self.bandmag(band, zpsys, time)


class RandomizedSupernova(Supernova):
    """Class for randomizing a supernova."""

    def __init__(
        self,
        sn_type,
        redshift,
        absolute_mag=None,
        absolute_mag_band="bessellb",
        mag_zpsys="AB",
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        random_seed=42,
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
        :param random_seed: Random seed for randomization
        :type random_seed: int
        """

        np.random.seed(random_seed)

        all_models, accepted_types = get_accepted_sn_types()
        if sn_type not in accepted_types:
            raise RuntimeError(
                "You passed %s as your SN type, " % sn_type
                + "but currently accepted SN types are: "
                + ", ".join(accepted_types)
            )
        self._sn_type = sn_type
        self._accepted_SN_types = accepted_types
        self._all_sncosmo_models = all_models
        self._type_models = None
        self._absolute_mag_band = absolute_mag_band

        self.set_random_sed_model(self._sn_type, random_seed=random_seed)

        if absolute_mag is None:
            absolute_mag = self.get_absolute_magnitude(
                self._sn_type, random_seed=random_seed
            )

        super(RandomizedSupernova, self).__init__(
            source=self._sncosmo_source,
            redshift=redshift,
            sn_type=self._sn_type,
            absolute_mag=absolute_mag,
            absolute_mag_band=absolute_mag_band,
            mag_zpsys=mag_zpsys,
            cosmo=cosmo,
            **kwargs
        )
        if self._sn_type == "Ia":
            self.set(**{"c": np.random.normal(0, 0.1), "x1": np.random.normal(0, 1)})

    def set_random_sed_model(self, sn_type, random_seed=42):
        """Function to set a random SED model for a given SN type.

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param random_seed: Random seed for randomization
        :type random_seed: int
        """
        np.random.seed(random_seed)

        if sn_type not in self._accepted_SN_types:
            raise RuntimeError(
                "You passed %s as your SN type, " % sn_type
                + "but currently accepted SN types are: "
                + ", ".join(accepted_types)
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

    def get_absolute_magnitude(
        self, sn_type, absolute_mag_distribution=None, random_seed=42
    ):
        """Function to get a reasonable absolute mag for a given SN type.

        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param absolute_mag_distribution: A function that returns an absolute mag
        :type absolute_mag_distribution: func
        :param random_seed: Random seed for randomization
        :type random_seed: int
        """

        np.random.seed(random_seed)

        if absolute_mag_distribution is None:
            mu, sigma = _ABSOLUTE_MAG_DISTS[sn_type]
            absolute_mag = np.random.normal(mu, sigma)
            self._absolute_mag_band = "bessellb"
        else:
            absolute_mag = absolute_mag_distribution()
        return absolute_mag


if __name__ == "__main__":

    test = RandomizedSupernova("IIP")
