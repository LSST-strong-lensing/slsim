import numpy as np
from slsim.Sources import random_supernovae
from astropy.table import Column, Table
from slsim.Sources.SourceVariability.variability import Variability
from slsim.Sources.SourceTypes.source_base import SourceBase

class Supernova(SourceBase):
    """A class to manage a supernova"""
    def __init__(self,
        source_dict,
        cosmo=None,
        **kwargs
        ):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This table or dict should contain atleast redshift of a supernova, offset from 
         the host if host galaxy is available. 
         eg: {"z": 0.8, "ra_off": 0.001, "dec_off": 0.005}
        :type source_dict: dict or astropy.table.Table
        :param cosmo: astropy.cosmology instance
        :param kwargs: dictionary of keyword arguments for a supernova. It sould contain
          following keywords:
            :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
            :type variability_model: str
            :param kwargs_variability: Keyword arguments for variability class.
            This is associated with an input for Variability class. By using these key
            words, code search for quantities in source_dict with these names and creates
            a dictionary and this dict should be passed to the Variability class.
            :type kwargs_variability: list of str
            :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
            :type sn_type: str
            :param sn_absolute_mag_band: Band used to normalize to absolute magnitude
            :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
            :param sn_absolute_zpsys: Optional, AB or Vega (AB default)
            :type sn_absolute_zpsys: str
            :param lightcurve_time: observation time array for lightcurve in unit of days.
            :type lightcurve_time: array
            :param sn_modeldir: sn_modeldir is the path to the directory containing files
            needed to initialize the sncosmo.model class. For example,
            sn_modeldir = 'C:/Users/username/Documents/SALT3.NIR_WAVEEXT'. These data can
            be downloaded from https://github.com/LSST-strong-lensing/data_public .
            For more detail, please look at the documentation of RandomizedSupernovae
            class.
            :type sn_modeldir: str
        """
        
        super().__init__(source_dict = source_dict)
        # These are the keywords that kwargs dict should contain
        self.cosmo = cosmo
        self.variability_model = kwargs.get("variability_model")
        self.kwargs_variability = kwargs.get("kwargs_variability")
        self.sn_type = kwargs.get("sn_type")
        self.sn_absolute_mag_band = kwargs.get("sn_absolute_mag_band")
        self.sn_absolute_zpsys = kwargs.get("sn_absolute_zpsys")
        self.lightcurve_time = kwargs.get("lightcurve_time")
        self.sn_modeldir = kwargs.get("sn_modeldir")
    @property
    def light_curve(self):
        if self.kwargs_variability is not None:
            # Here we extract lightcurves of a supernova in given bands
            kwargs_variab_extracted = {}
            z = self.source_dict["z"]
            if self.cosmo is None:
                raise ValueError(
                    "Cosmology cannot be None for Supernova class. Please"
                    "provide a suitable astropy cosmology."
                )
            else:
                lightcurve_class = random_supernovae.RandomizedSupernova(
                    sn_type=self.sn_type,
                    redshift=z,
                    absolute_mag=None,
                    absolute_mag_band=self.sn_absolute_mag_band,
                    mag_zpsys=self.sn_absolute_zpsys,
                    cosmo=self.cosmo,
                    modeldir=self.sn_modeldir,
                )

            for element in list(self.kwargs_variability):
                # if lsst filter is being used
                if element in [
                    "r",
                    "i",
                    "g",
                    "z",
                    "y",
                    "F062",
                    "F087",
                    "F106",
                    "F129",
                    "F158",
                    "F184",
                    "F146",
                    "F213",
                ]:
                    if element in ["r", "i", "g", "z", "y"]:
                        provided_band = "lsst" + element
                    else:
                        provided_band = element
                    name = "ps_mag_" + element
                    times = self.lightcurve_time
                    magnitudes = lightcurve_class.get_apparent_magnitude(
                        time=times,
                        band=provided_band,
                        zpsys=self.sn_absolute_zpsys,
                    )
                    new_column = Column([float(min(magnitudes))], name=name)
                    self._source_dict = Table(self.source_dict)
                    if name not in self._source_dict.colnames:
                        self._source_dict.add_column(new_column)
                        self.source_dict = self._source_dict[0]
                    kwargs_variab_extracted[element] = {
                        "MJD": times,
                        name: magnitudes,
                    }
        else:
            kwargs_variab_extracted = None
        return kwargs_variab_extracted
    
    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """

        if not hasattr(self, "kwargs_variab_dict"):
            self.kwargs_variab_dict = self.light_curve
        column_names = self.source_dict.colnames
        if "ps_mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "ps_mag_" + band
        if self.kwargs_variab_dict is not None:
            if band in self.kwargs_variab_dict.keys():
                kwargs_variab_band = self.kwargs_variab_dict[band]
            else:
                kwargs_variab_band = self.kwargs_variab_dict
            self.variability_class = Variability(
                self.variability_model, **kwargs_variab_band
            )
        else:
            self.variability_class = None
        if image_observation_times is not None:
            if self.variability_class is not None:
                variable_mag = self.variability_class.variability_at_time(
                    image_observation_times
                )
                return variable_mag
            else:
                raise ValueError(
                    "variability model is not provided. Please include"
                    "one of the variability models in your kwargs_variability."
                )
        else:
            source_mag = self.source_dict[band_string]
            if (
                isinstance(source_mag, np.ndarray)
                and source_mag.ndim == 2
                and source_mag.shape[0] == 1
            ):
                return source_mag.reshape(-1)
            else:
                return source_mag
    
