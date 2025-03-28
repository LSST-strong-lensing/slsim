from abc import ABC
import numpy as np
from astropy.table import Table

class PointSourceBase(ABC):
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

        # If center_x and center_y are already specified, use them.
        if (
            "center_x" in self.source_dict.colnames
            and "center_y" in self.source_dict.colnames
        ):
            self._center_source = np.array(
                [self.source_dict["center_x"], self.source_dict["center_y"]]
            )
        else:
            self._center_source = [None, None]
    
    @property
    def extended_source_position(self):
        return self._center_source

    @property
    def point_source_position(self):
        """Point source position. point source could be at the center of the
        extended source or it can be off from center of the extended source.
        
        :return: [x_pos, y_pos]
        """

        if self._center_source[0] is not None:
            if "ra_off" in self.source_dict.colnames:
                center_x_point_source = self._center_source[0] + float(
                    self.source_dict["ra_off"]
                )
                center_y_point_source = self._center_source[1] + float(
                    self.source_dict["dec_off"]
                )
                self._center_point_source = np.array(
                    [center_x_point_source, center_y_point_source]
                )
            else:
                self._center_point_source = self._center_source
            return self._center_point_source
        return self._center_source
    
    @property
    def redshift(self):
        """Returns source redshift."""
        if isinstance(self.source_dict["z"], list):
            return float(self.source_dict["z"][0])
        return float(self.source_dict["z"])



