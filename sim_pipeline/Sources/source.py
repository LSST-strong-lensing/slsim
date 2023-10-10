import numpy as np

class Source(object):
    """This class provides source dictionary and variable magnitude of a 
    individual source.
    """
    def __init__(self, source_dict, magnification = None, variability_model = None, 
                 kwargs_variab = None, image_observation_times = None):
        """
        :param source_dict: Source properties
        :type source_dict: dict
        :param magnification: Array of lensing magnification of each images. If None, 
         considers unlensed case
        :param variability_model: Variability model for source.
        :param kwargs_variab: Keyword arguments for variability class.
        :param image_observation_times: Images observation time for each image. Eg: for 
         two images, the form of this param is np.array([[t11, t12,...t1n],
         [t21, t22,...t2n]]). This should be based on lensing calculation.
        """
        self.source_dict = source_dict
        self.magnification = magnification
        self.variability_model = variability_model
        self.kwargs_variab = kwargs_variab
        self.image_observation_times = image_observation_times

    def magnitude(self, band):
        """
        Get the magnitude of the source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the source in the specified band
        :rtype: float
        """

        band_string = "mag_" + band
        if self.magnification is not None:
            mag = self.magnification
            source_mag = self.source_dict[band_string] - 2.5 * np.log10(np.abs(mag))
            if self.variability_model is not None:
                from sim_pipeline.Sources.source_variability.variability \
                    import Variability
                kwargs_variability = self.kwargs_variab
                variability_class = Variability(**kwargs_variability)
                if self.variability_model == "sinusoidal":
                    function = variability_class.sinusoidal_variability
                else:
                    raise ValueError(
                        "given model is not supported. Currently,"
                        "supported model is sinusoudal."
                    )
                observed_time = self.image_observation_times
                variable_mag_array = []
                for i in range(len(source_mag)):
                    for j in range(len(observed_time[i])):
                        variable_mag_array.append(
                            source_mag[i] + function(observed_time[i][j])
                        )
                variable_mag = np.array(variable_mag_array).reshape(len(source_mag), 
                                                                len(observed_time[0]))
                return variable_mag
            else:
                return source_mag
        else:
            source_mag = self.source_dict[band_string]
            return source_mag
        
    def to_dict(self):
        """
        Convert the Source object to a dictionary.

        :return: Dictionary representation of the individual source
        :rtype: dict
        """
        return self.source_dict