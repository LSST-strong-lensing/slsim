import numpy as np
import numpy.random as random
import sncosmo
from slsim.Util import param_util



class Supernovae(sncosmo.Model):
    """Class describing elliptical galaxies."""

    def __init__(self, source, redshift, sn_type, absolute_mag, 
                        absolute_mag_band, absolute_mag_zpsys):
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
        :param absolute_mag_zpsys: AB or Vega
        :type absolute_mag_zpsys: str
        """
        super(Supernovae, self).__init__(source=source)
        
        self._parameters[0] = redshift
        self._sn_type = sn_type
        self.set_source_peakabsmag(absolute_mag,absolute_mag_band,absolute_mag_zpsys)

        




    