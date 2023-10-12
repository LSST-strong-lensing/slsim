import numpy as np


class Source(object):
    """This class provides source dictionary and variable magnitude of a individual
    source."""

    def __init__(self, source_dict, kwargs_variab=None):
        """
        :param source_dict: Source properties
        :type source_dict: dict
        :param variability_model: Variability model for source.
        :param kwargs_variab: Keyword arguments for variability class.
        """
        self.source_dict = source_dict
        if kwargs_variab is not None:
            self.variability_model = kwargs_variab["variability_model"]
            self.kwargs_variability_model = {
                key: value
                for key, value in kwargs_variab.items()
                if key != list(kwargs_variab.keys())[0]
            }
        else:
            self.variability_model = None
            self.kwargs_variability_model = None

    def magnitude(self, band, magnification=None, image_observation_times=None):
        """Get the magnitude of the source in a specific band.

        :param band: Imaging band
        :type band: str
        :param magnification: Array of lensing magnification of each images. If None,
         considers unlensed case.
        :param image_observation_times: Images observation time for each image. Eg: for
         two images, the form of this param is np.array([[t11, t12,...t1n],
         [t21, t22,...t2n]]). This should be based on lensing calculation.
        :return: Magnitude of the source in the specified band
        :rtype: float
        """
        self.magnification = magnification
        self.image_observation_times = image_observation_times

        band_string = "mag_" + band
        if self.magnification is not None:
            mag = self.magnification
            source_mag = self.source_dict[band_string] - 2.5 * np.log10(np.abs(mag))
            if self.image_observation_times is not None:
                if self.variability_model is not None:
                    from sim_pipeline.Sources.source_variability.variability import (
                        Variability,
                    )
                else:
                    raise ValueError(
                        "variability model is not provided. Please include one of the"
                        "variability models in your kwargs_variability as a first entry."
                    )
                kwargs_variability_model = self.kwargs_variability_model
                variability_class = Variability(**kwargs_variability_model)
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
                variable_mag = np.array(variable_mag_array).reshape(
                    len(source_mag), len(observed_time[0])
                )
                return variable_mag
            else:
                return source_mag
        else:
            source_mag = self.source_dict[band_string]
            return source_mag

    def to_dict(self):
        """Convert the Source object to a dictionary.

        :return: Dictionary representation of the individual source
        :rtype: dict
        """
        return self.source_dict
