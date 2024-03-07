import sncosmo
from astropy import cosmology


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

        :param time: The observer-frame time array to evaluate the model (in days)
        :type time: `~np.ndarray` or list
        :param band: The bandpass to evaluate the model over
        :type band: str or `~sncosmo.Bandpass`
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str

        :return: magnitude of source
        """

        return self.bandmag(band, zpsys, time)
