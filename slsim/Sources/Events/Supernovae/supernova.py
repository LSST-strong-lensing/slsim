import os
from warnings import warn
import sncosmo
from sncosmo.bandpasses import get_bandpass
import numpy as np
from astropy import cosmology

# Reading a sncosmo source from disk dominates the cost of building a
# Supernova and depends only on the model, not on the individual supernova, so a
# population loop can pay it once instead of once per object.
_SOURCE_CACHE = {}


def _source_from_modeldir(modeldir, sn_type, source):
    """Build an sncosmo source from a model directory, reusing a cached
    template when the same model has already been read from disk.

    :param modeldir: directory including files for supernova models
    :type modeldir: str
    :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
    :type sn_type: str
    :param source: name of the SED model, ignored for type Ia
    :type source: str
    :return: `~sncosmo.Source`, template shared by every supernova using this model
    """
    key = (modeldir, sn_type, source)
    if key not in _SOURCE_CACHE:
        if sn_type == "Ia":
            _SOURCE_CACHE[key] = sncosmo.SALT3Source(modeldir=modeldir)
        else:
            sed_file = os.path.join(modeldir, sn_type, source) + ".SED"
            phase, wave, flux = sncosmo.read_griddata_ascii(sed_file)
            _SOURCE_CACHE[key] = sncosmo.TimeSeriesSource(
                phase=phase, wave=wave, flux=flux
            )
    return _SOURCE_CACHE[key]


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
            source = _source_from_modeldir(modeldir, sn_type, source)

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

        :return: magnitude of source. Never NaN: times at which the supernova has
            no flux, and bands the SED model does not cover, give inf.
        """
        bandpass = get_bandpass(band)

        scalar_time = np.ndim(time) == 0
        time_array = np.atleast_1d(np.asarray(time, dtype=float))

        # Infinite magnitude, i.e. no flux, wherever the supernova is not visible.
        magnitude = np.full(time_array.shape, np.inf)

        if bandpass.minwave() < self.minwave() or bandpass.maxwave() > self.maxwave():
            warn(
                "no flux assigned in bandpass {0!r:s}: it lies outside the spectral "
                "range of the supernova model. Use extended wavelength SN models "
                "found here: https://github.com/LSST-strong-lensing/data_public/tree/main/sncosmo_sn_models".format(
                    bandpass.name
                )
            )
            return magnitude[0] if scalar_time else magnitude

        # The template has zero flux outside its phase range, so only integrate over
        # the times it covers; observation windows are typically much longer than
        # the supernova itself.
        covered = (time_array >= self.mintime()) & (time_array <= self.maxtime())
        if covered.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                magnitude[covered] = self.bandmag(bandpass, zpsys, time_array[covered])
            # At high redshift a band probes the far ultraviolet, where the template
            # flux can be zero or negative across the whole band and bandmag returns
            # inf or NaN. Both mean no flux.
            magnitude[np.isnan(magnitude)] = np.inf

        if self._sn_type != "Ia":
            # This line is needed because non type Ia supernovae lightcurves do not drop to
            # zero flux as they should
            magnitude = np.where(time_array > self.source.minphase(), magnitude, 10**8)

        if scalar_time:
            return magnitude[0]
        return magnitude

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
