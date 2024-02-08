from slsim.Sources.SourceVariability.variability import (
    Variability,
)
import numpy as np


class Source(object):
    """This class provides source dictionary and variable magnitude of a individual
    source."""

    def __init__(self, source_dict, variability_model=None, kwargs_variability=None):
        """
        :param source_dict: Source properties
        :type source_dict: dict
        :param variability_model: keyword for variability model to be used. This is an
         input for the Variability class.
        :type variability_model: str
        :param kwargs_variability: Keyword arguments for variability class.
         This is associated with an input for Variability class. By using these key
         words, code search for quantities in source_dict with these names and creates
         a dictionary and this dict should be passed to the Variability class.
        :type kwargs_variability: list of str
        """
        self.source_dict = source_dict
        if kwargs_variability is not None:
            kwargs_variab_extracted = {}
            for element in kwargs_variability:
                if element in self.source_dict.colnames:
                    if (
                        isinstance(self.source_dict[element], np.ndarray)
                        and self.source_dict[element].ndim == 2
                        and self.source_dict[element].shape[0] == 1
                    ):
                        kwargs_variab_extracted[element] = self.source_dict[
                            element
                        ].reshape(-1)
                    else:
                        kwargs_variab_extracted[element] = self.source_dict[element]
                else:
                    raise ValueError(
                        "given keyword %s is not in the source catalog." % element
                    )
            self.variability_class = Variability(
                variability_model, **kwargs_variab_extracted
            )
        else:
            self._variability_class = None

    @property
    def redshift(self):
        """Returns source redshift."""

        return self.source_dict["z"]

    @property
    def n_sersic(self):
        """Returns sersic index of the source."""

        return self.source_dict["n_sersic"]

    @property
    def angular_size(self):
        """Returns angular size of the source."""

        return self.source_dict["angular_size"]

    @property
    def ellipticity(self):
        """Returns ellipticity components of source."""

        return self.source_dict["e1"], self.source_dict["e2"]

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        column_names = self.source_dict.colnames
        if "ps_mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "ps_mag_" + band

        # source_mag = self.source_dict[band_string]
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

    def extended_source_magnitude(self, band):
        """Get the magnitude of the extended source in a specific band.

        :param band: Imaging band
        :type band: str
        :return: Magnitude of the extended source in the specified band
        :rtype: float
        """
        column_names = self.source_dict.colnames
        if "mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "mag_" + band
        source_mag = self.source_dict[band_string]
        return source_mag

    def extended_source_position(self, deflector_center, test_area):
        """Extended source position. If not present from the catalog, it is drawn
        uniformly within the circle of the test area centered on the deflector position.

        :param deflector_center: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :return: [x_pos, y_pos]
        """
        self.center_lens = deflector_center
        self.test_area = test_area

        if not hasattr(self, "_center_source"):
            # Define the radius of the test area circle
            test_area_radius = np.sqrt(self.test_area / np.pi)
            # Randomly generate a radius within the test area circle
            r = np.sqrt(np.random.random()) * test_area_radius
            theta = 2 * np.pi * np.random.random()
            # Convert polar coordinates to cartesian coordinates
            center_x_source = self.center_lens[0] + r * np.cos(theta)
            center_y_source = self.center_lens[1] + r * np.sin(theta)
            self._center_source = np.array([center_x_source, center_y_source])
        return self._center_source

    def point_source_position(self, deflector_center, test_area):
        """Point source position. point source could be at the center of the extended
        source or it can be off from center of the extended source.

        :param deflector_center: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :return: [x_pos, y_pos]
        """
        self.center_lens = deflector_center
        self.test_area = test_area
        extended_source_center = self.extended_source_position(
            self.center_lens, self.test_area
        )

        if "ra_off" in self.source_dict.colnames:
            center_x_point_source = (
                extended_source_center[0] + self.source_dict["ra_off"][0]
            )
            center_y_point_source = (
                extended_source_center[1] + self.source_dict["dec_off"][0]
            )
            self._center_point_source = np.array(
                [center_x_point_source, center_y_point_source]
            )
            return self._center_point_source
        return extended_source_center
