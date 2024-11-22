import os
from warnings import warn
import sncosmo
from sncosmo.bandpasses import get_bandpass
import numpy as np
from astropy import cosmology


class Supernova(sncosmo.Model):
    """Class for initializing a supernova of the type sn_type specified by the
    user. If modeldir is provided by the user and sn_type is Ia, the
    sncosmo.SALT3Source class is first used to model the supernova. In this
    case, modeldir is the path to the directory containing files needed to
    initialize this class. For example, modeldir =
    'C:/Users/username/Documents/SALT3.NIR_WAVEEXT' Afterwards, this
    sncosmo.SALT3Source class is passed into the sncosmo.Model class. If
    modeldir is provided by the user and sn_type is other than Ia, the
    sncosmo.TimeSeriesSource class is used to model the supernova. In this
    case, modeldir is the path to the full list of models. For example,
    modeldir = 'C:/Users/username/Documents/NON1ASED.V19_CC+HostXT_WAVEEXT
    Afterwards, this sncosmo.TimeSeriesSource class is passed into the
    sncomsmo.Model class.

    These files can be found in
    https://github.com/LSST-strong-lensing/data_public
    If    modeldir is not provided by the user, the sncosmo.Models class
    is directly used to    model the supernova by retrieving the
    specified sn model from sncosmo's list of    built-in models, which
    can be found here:
    https://sncosmo.readthedocs.io/en/stable/source-list.html
    """

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
        modeldir=None,
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
        :param modeldir: directory including files for supernova models
        :type modeldir: str
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

        self._sn_type = sn_type
        if modeldir is not None:
            if sn_type == "Ia":
                source = sncosmo.SALT3Source(
                    modeldir=modeldir,
                )
            else:
                modeldir = os.path.join(modeldir, sn_type, source) + ".SED"
                phase, wave, flux = sncosmo.read_griddata_ascii(modeldir)
                source = sncosmo.TimeSeriesSource(
                    phase=phase,
                    wave=wave,
                    flux=flux,
                )

        super(Supernova, self).__init__(source=source, **kwargs)
        self._parameters[0] = redshift
        self.set_source_amplitude(
            absolute_mag,
            absolute_mag_band,
            peak_apparent_mag,
            peak_apparent_mag_band,
            mag_zpsys,
            cosmo=cosmo,
        )

    def get_apparent_magnitude(self, time, band, zpsys="AB"):
        """Function to return apparent magnitude of a SN for a given band and
        time.

        :param time: The observer-frame time array to evaluate the model (in days)
        :type time: `~np.ndarray` or list
        :param band: The bandpass to evaluate the model over
        :type band: str or `~sncosmo.Bandpass`
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str

        :return: magnitude of source
        """
        bandpass = get_bandpass(band)

        if bandpass.minwave() < self.minwave() or bandpass.maxwave() > self.maxwave():
            warn(
                "bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] "
                "outside spectral range [{3:.6g}, .., {4:.6g}]\n"
                "Ignoring bandpass for now. Use extended wavelength SN models "
                "found here: https://github.com/LSST-strong-lensing/data_public/tree/main/sncosmo_sn_models".format(
                    bandpass.name,
                    bandpass.minwave(),
                    bandpass.maxwave(),
                    self.minwave(),
                    self.maxwave(),
                )
            )
            return np.ones_like(time) * np.NaN

        minphase = self.source.minphase()
        if self._sn_type == "Ia":
            return self.bandmag(band, zpsys, time)
        else:
            # This line is needed because non type Ia supernovae lightcurves do not drop to
            # zero flux as they should
            return np.where(time > minphase, self.bandmag(band, zpsys, time), 10**8)

    def set_source_amplitude(
        self,
        absmag,
        abs_mag_band,
        peak_apparent_mag,
        peak_apparent_mag_band,
        magsys,
        cosmo,
    ):
        """Sets the amplitude of the source component of the model according to
        the desired absolute magnitude in the specified band.

        If the absolute magnitude is not given, then sets the amplitude of the source
        component of the model according to a peak apparent magnitude.

        If neither the absolute magnitude nor apparent magnitude are given, a warning
        message is displayed.

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

        :return: Nothing is returned. The source's amplitude parameter is modified in place.
        """
        if absmag is not None:
            if abs_mag_band is None:
                raise Exception(
                    "Must set absolute_mag_band when attempting to set an absolute magnitude."
                )
            else:
                self.set_source_peakabsmag(absmag, abs_mag_band, magsys, cosmo=cosmo)
                if peak_apparent_mag is not None:
                    print(
                        "Both peak_apparent_mag and absolute_mag were given, choosing absolute_mag."
                    )
        elif peak_apparent_mag is not None:
            self.set_source_peakmag(peak_apparent_mag, peak_apparent_mag_band, magsys)
        else:
            warn(
                "Use self.set_source_peakabsmag or sefl.set_peakmag to set the amplitude."
            )
