from abc import ABC
import numpy as np
from astropy.table import Table

class SourceBase(ABC):
    """Class of a single source with quantities only related to the
    source (independent of the deflector)"""
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
        ):

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
        self.source_dict = source_dict
        self.variability_model = variability_model
        self.kwargs_variability = kwargs_variability
        self.sn_type = sn_type
        self.sn_absolute_mag_band = sn_absolute_mag_band
        self.sn_absolute_zpsys = sn_absolute_zpsys
        self.cosmo = cosmo
        self.lightcurve_time = lightcurve_time
        self.sn_modeldir = sn_modeldir

    def source_position(self, center_lens, draw_area):
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
