from slsim.Sources.SourceTypes.supernova_event import SupernovaEvent
from slsim.Sources.SourceTypes.quasar import Quasar

_SUPPORTED_POINT_SOURCES = ["supernova", "quasar"]

class PointSource(object):
    """class to manage a single point source"""
    def __init__(self,
        source_dict,
        cosmo=None,
        **kwargs
        ):
        """One can supply either supernovae kwargs or agn kwargs. 
         If supernovae kwargs are supplied, agn kwrgs can be None which is a 
         default option.

        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         For more detail, please see documentation of SupernovaEvent and Quasar class.
        :type source_dict: dict or astropy.table.Table
        :param cosmo: astropy.cosmology instance
        :param pointsource_type: keyword for specifying point source type.
        :type pointsource_type: str. supported types are "supernova", "quasar".
        :param kwargs: dictionary of keyword arguments for a point source. It should 
         contain keywords for pointsource_type and other keywords associated with 
         supernova and quasar. For supernova kwargs dict, please see documentation of 
         SupernovaEvent class. For quasar kwargs dict, please see documentation of 
         Quasar class.
         Eg of supernova kwargs: kwargs={"pointsource_type": "supernova", 
          "variability_model": "light_curve", "kwargs_variability": ["supernovae_lightcurve",
            "i", "r"], "sn_type": "Ia", "sn_absolute_mag_band": "bessellb", 
            "sn_absolute_zpsys": "ab", "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": "/Users/narayankhadka/Downloads/sncosmo_sn_models/SALT3.NIR_WAVEEXT/"}.
         Other supported pointsource_types are "supernova", "quasar".
        """
        self.pointsource_type = kwargs["pointsource_type"]
        if self.pointsource_type in ["supernova"]:
            self._point_source = SupernovaEvent(source_dict=source_dict, cosmo=cosmo, **kwargs)
        elif self.pointsource_type in ["quasar"]:
            self._point_source = Quasar(source_dict=source_dict, cosmo=cosmo, **kwargs)
        else:
            raise ValueError(
                "Point source type %s not supported. Chose among %s."
                % (self.pointsource_type, _SUPPORTED_POINT_SOURCES)
            )
        
    @property
    def redshift(self):
        """Returns source redshift."""

        return self._point_source.redshift
    
    @property
    def point_source_offset(self):
        """Provides point source offset from host center"""
        
        return self._point_source.point_source_offset
    
    @property
    def point_source_position(self):
        """Point source position. point source could be at the center of the
        extended source or it can be off from center of the extended source.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        return self._point_source.point_source_position
    
    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        return self._point_source.point_source_magnitude(band=band, 
                    image_observation_times=image_observation_times)