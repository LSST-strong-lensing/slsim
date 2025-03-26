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
        agn_known_band=None,
        agn_known_mag=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        source_type=None,
        light_profile=None,
        ):

        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
        :type source_dict: dict or astropy.table.Table
        When source_type is 'interpolated', include keys:
        - 'z' (float)
        - 'image' (numpy.ndarray)
        - 'z_data' (float)
        - 'pixel_width_data' (float)
        - 'phi_G' (float)
        - 'center_x' (float)
        - 'center_y' (float)
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
        :param source_type: type of the source 'extended' or 'point_source' or
         'point_plus_extended' supported
        :type source_type: str
        :param light_profile: keyword for number of sersic profile to use in source
         light model
        :type light_profile: str . Either "single_sersic", "double_sersic", or "interpolated" .
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
        self.variability_model = variability_model
        self.kwargs_variability = kwargs_variability
        self.sn_type = sn_type
        self.sn_absolute_mag_band = sn_absolute_mag_band
        self.sn_absolute_zpsys = sn_absolute_zpsys
        self.cosmo = cosmo
        self.lightcurve_time = lightcurve_time
        self.sn_modeldir = sn_modeldir
        self.agn_known_band = agn_known_band
        self.agn_known_mag = agn_known_mag
        self.agn_driving_variability_model = agn_driving_variability_model
        self.agn_driving_kwargs_variability = agn_driving_kwargs_variability
        self.source_type = source_type
        self.light_profile = light_profile

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

        return float(self.source_dict["z"])
