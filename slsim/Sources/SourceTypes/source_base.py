from abc import ABC
import numpy as np
from astropy.table import Table

class SourceBase(ABC):
    """Class of a single source with quantities only related to the
    source (independent of the deflector)"""
    def __init__(self, 
        source_dict
        ):

        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
        :type source_dict: dict or astropy.table.Table
        """

        # Convert dict to astropy table
        if isinstance(source_dict, dict):
            self.source_dict = Table([source_dict])[0]
        else:  # if source_dict is already an astropy table
            self.source_dict = source_dict

        # If center_x and center_y are already specified, use them instead of picking randomly
        if (
            "center_x" in self.source_dict.colnames
            and "center_y" in self.source_dict.colnames
        ):
            self._center_source = np.array(
                [self.source_dict["center_x"], self.source_dict["center_y"]]
            )

    def extended_source_position(self, center_lens, draw_area):
        """source position. If a center has already been provided (and
        stored in self._center_source during initialization), then it is simply
        returned. Otherwise, a source position is drawn uniformly within the
        circle of the test area centered on the deflector position.

        :param center_lens: center of the deflector.
            Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a source
            position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """
        if hasattr(self, "_center_source"):
            return self._center_source

        test_area_radius = np.sqrt(draw_area / np.pi)
        r = np.sqrt(np.random.random()) * test_area_radius
        theta = 2 * np.pi * np.random.random()
        self._center_source = np.array(
            [center_lens[0] + r * np.cos(theta), center_lens[1] + r * np.sin(theta)]
        )
        return self._center_source
    
    def point_source_position(self, center_lens, draw_area):
        """Point source position. point source could be at the center of the
        extended source or it can be off from center of the extended source. In
        the absence of a point source, this is the center of the extended
        source.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        # This is a extended source center which will be used to determine point 
        # source center. if point source offset is not given, this will be the center 
        # of the point source too.
        source_center = self.extended_source_position(center_lens, draw_area)

        if "ra_off" in self.source_dict.colnames:
            center_x_point_source = source_center[0] + float(
                self.source_dict["ra_off"]
            )
            center_y_point_source = source_center[1] + float(
                self.source_dict["dec_off"]
            )
            self._center_point_source = np.array(
                [center_x_point_source, center_y_point_source]
            )
            return self._center_point_source
        return source_center
    
    @property
    def redshift(self):
        """Returns source redshift."""
        if isinstance(self.source_dict["z"], list):
            return float(self.source_dict["z"][0])
        return float(self.source_dict["z"])
