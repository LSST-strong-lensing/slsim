from slsim.Sources.source_variability.variability import (
                        Variability,
                    )


class Source(object):
    """This class provides source dictionary and variable magnitude of a individual
    source."""

    def __init__(self, source_dict, variability_model=None, kwargs_variab = None):
        """
        :param source_dict: Source properties
        :type source_dict: dict
        :param variability_model: Variability model for source.
        :param kwargs_variab: Keyword arguments for variability class.
        """
        self.source_dict = source_dict
        self._variability_model = variability_model
        self._kwargs_variab = kwargs_variab
        if isinstance(self._kwargs_variab, dict):
            dictionary = True
        else:
            dictionary = False
        if self._kwargs_variab is not None:
            if dictionary is True:
                self._variability_class = Variability(variability_model,**kwargs_variab)
            else:
                kwargs_variab_extracted = {}
                for element in self._kwargs_variab:
                    if element in self.source_dict.colnames:
                        kwargs_variab_extracted[element]=self.source_dict[element]
                        self._variability_class = Variability(variability_model
                                                            ,**kwargs_variab_extracted)
                    else:
                        raise ValueError("given keywords are not in provided the"
                                          "source catalog")
        else:
            self._variability_class = None
    @property    
    def redshift(self):
        """returns source redshift."""

        return self.source_dict["z"]
    
    @property 
    def n_sersic(self):
        """returns sersic index of the source"""

        return self.source_dict["n_sersic"]
    
    @property 
    def angular_size(self):
        """returns angular size of the source"""

        return self.source_dict["angular_size"]
    
    @property 
    def ellipticity_1(self):
        """returns 1st ellipticity component of source"""

        return self.source_dict["e1"]
    
    @property
    def ellipticity_2(self):
        """returns 2nd ellipticity component of source"""

        return self.source_dict["e2"]


    def magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the source in a specific band.

        :param band: Imaging band
        :type band: str
        :param magnification: Array of lensing magnification of each images. If None,
         considers unlensed case.
        :param image_observation_times: Images observation time for an image.
        :return: Magnitude of the source in the specified band
        :rtype: float
        """

        band_string = "mag_" + band

        #source_mag = self.source_dict[band_string]
        if image_observation_times is not None:
            if self._variability_class is not None:
                variable_mag = self.source_dict[band_string] + \
                      self._variability_class.variability_at_t(image_observation_times)
                return variable_mag
            else:
                raise ValueError(
                    "variability model is not provided. Please include"
                    "one of the variability models in your kwargs_variability."
                )
        else:
            source_mag = self.source_dict[band_string]
            return source_mag
