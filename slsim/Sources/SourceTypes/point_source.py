from slsim.Sources.SourceTypes.supernova import Supernova
from slsim.Sources.SourceTypes.quasar import Quasar

_SUPPORTED_LIGHT_CURVES = ["supernovae_lightcurve", "agn_lightcurve"]

class PointSource(object):
    """class to manage a single point source"""
    def __init__(self,
        source_dict,
        variability_model=None,
        kwargs_variability=None,
        sn_type=None,
        sn_absolute_mag_band=None,
        sn_absolute_zpsys=None,
        cosmo=None,
        lightcurve_time=None,
        sn_modeldir=None,
        agn_known_band=None,
        agn_known_mag=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        ):
        """One can supply either supernovae kwargs or agn kwargs. 
         If supernovae kwargs are supplied, agn kwrgs can be None which is a 
         default option.

        :param source_dict: Source properties. May be a dictionary or an Astropy table.
        :type source_dict: dict or astropy.table.Table
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
        :param agn_known_band: Speclite filter of which the magnitude is known. Used to normalize
         mean magnitudes.
        :type agn_known_band: str
        :param agn_known_mag: Magnitude of the agn in the known band.
        :type agn_known_mag: float
        :param agn_driving_variability_model: Variability model with light_curve output
         which drives the variability across all bands of the agn.
        :type agn_driving_variability_model: str (e.g. "light_curve", "sinusoidal", "bending_power_law")
        :param agn_driving_kwargs_variability: Dictionary containing all variability parameters
         for the driving variability class
        :type agn_driving_kwargs_variability: dict
        """

        if "supernovae_lightcurve" in kwargs_variability:
            self._source = Supernova(source_dict=source_dict,
                            variability_model=variability_model,
                            kwargs_variability=kwargs_variability,
                            sn_type=sn_type,
                            sn_absolute_mag_band=sn_absolute_mag_band,
                            sn_absolute_zpsys=sn_absolute_zpsys,
                            cosmo=cosmo,
                            lightcurve_time=lightcurve_time,
                            sn_modeldir=sn_modeldir,)
        elif "agn_lightcurve" in kwargs_variability:
            self._source = Quasar(source_dict=source_dict,
                            variability_model=variability_model,
                            kwargs_variability=kwargs_variability,
                            lightcurve_time=lightcurve_time,
                            agn_known_band=agn_known_band,
                            agn_known_mag=agn_known_mag,
                            agn_driving_variability_model=agn_driving_variability_model,
                            agn_driving_kwargs_variability=agn_driving_kwargs_variability,)
        else:
            raise ValueError(
                "Given kwargs_variability is not supported. Choose among %s."
                % _SUPPORTED_LIGHT_CURVES
            )
        
    @property
    def redshift(self):
        """Returns source redshift."""

        return self._source.redshift
    
    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        return self._source.point_source_magnitude(band=band, 
                    image_observation_times=image_observation_times)
