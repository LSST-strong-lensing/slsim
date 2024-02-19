from slsim.Sources.SourceVariability.variability import (
    Variability,
)
import numpy as np
from lenstronomy.Util import constants
from slsim.Sources.simple_supernova_lightcurve import SimpleSupernovaLightCurve
from astropy.table import Column, Table

class Source(object):
    """This class provides source dictionary and variable magnitude of a individual
    source."""

    def __init__(self, source_dict, variability_model=None, 
                 kwargs_variability=None, kwargs_peak_mag = None, cosmo=None,
                 lightcurve_time = None):
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
        :param kwargs_peak_mag: range of the peak magnitude
        :type kwargs_peak_mag: dict. eg: kwargs_peak_mag={"peak_mag_min": m_min, 
         "peak_mag_max": m_max}
        :param lightcurve_time: time period for lightcurve.
        :type lightcurve_time: astropy unit object. egs: 10*u.day, 10*u.year.
        """
        self.source_dict = source_dict
        if kwargs_variability is not None:
            kwargs_variab_extracted = {}
            kwargs_variability_list = ["absolute_magnitude", "peak_apparent_magnitude", 
                                       "r", "g", "i"]
            if any(element in kwargs_variability_list for
                    element in list(kwargs_variability)):
                lightcurve_class = SimpleSupernovaLightCurve(cosmo=cosmo)
                if kwargs_peak_mag is not None:
                    abs_mag=round(np.random.uniform(kwargs_peak_mag["peak_mag_min"],
                                                 kwargs_peak_mag["peak_mag_max"]), 5)
                else:
                    raise ValueError(
                            "kwargs_peak_mag is None. Please provide kwargs_peak_mag."
                        )
                provided_band = [element for element in 
                            list(kwargs_variability) if element in ['r', 'i', 'g']][0]
                times, magnitudes = lightcurve_class.generate_light_curve(
                    redshift=self.redshift, absolute_magnitude=abs_mag, 
                    lightcurve_time=lightcurve_time, band=provided_band)
                new_column = Column([float(min(magnitudes))], name="ps_mag_"+provided_band)
                self._source_dict = Table(self.source_dict)
                self._source_dict.add_column(new_column)
                self.source_dict = self._source_dict[0]
                kwargs_variab_extracted["MJD"]=times
                kwargs_variab_extracted["ps_mag_"+provided_band] = magnitudes
            else:
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

    def extended_source_position(self, center_lens, draw_area):
        """Extended source position. If not present from the catalog, it is drawn
        uniformly within the circle of the test area centered on the deflector position.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        if not hasattr(self, "_center_source"):
            # Define the radius of the test area circle
            test_area_radius = np.sqrt(draw_area / np.pi)
            # Randomly generate a radius within the test area circle
            r = np.sqrt(np.random.random()) * test_area_radius
            theta = 2 * np.pi * np.random.random()
            # Convert polar coordinates to cartesian coordinates
            center_x_source = center_lens[0] + r * np.cos(theta)
            center_y_source = center_lens[1] + r * np.sin(theta)
            self._center_source = np.array([center_x_source, center_y_source])
        return self._center_source

    def point_source_position(self, center_lens, draw_area):
        """Point source position. point source could be at the center of the extended
        source or it can be off from center of the extended source. In the absence of a
        point source, this is the center of the extended source.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :return: [x_pos, y_pos]
        """

        extended_source_center = self.extended_source_position(center_lens, draw_area)

        if "ra_off" in self.source_dict.colnames:
            center_x_point_source = extended_source_center[0] + float(
                self.source_dict["ra_off"]
            )
            center_y_point_source = extended_source_center[1] + float(
                self.source_dict["dec_off"]
            )
            self._center_point_source = np.array(
                [center_x_point_source, center_y_point_source]
            )
            return self._center_point_source
        return extended_source_center
    
    def kwargs_extended_source_light(self, center_lens, draw_area, band=None, 
                                     sersic_profile_str="single"):
        """ Provids dictionary of keywords for the source light model(s).

        :param band: Imaging band
        :param sersic_profile_str: number of sersic_profile
        :type sersic_profile_str: str . eg: "single" or "double".
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position(
        center_lens=center_lens, draw_area=draw_area
                )
        if sersic_profile_str == "single":
            size_source_arcsec = float(self.angular_size) / constants.arcsec
            kwargs_extended_source = [
                        {
                            "magnitude": mag_source,
                            "R_sersic": size_source_arcsec,
                            "n_sersic": float(self.n_sersic),
                            "e1": float(self.ellipticity[0]),
                            "e2": float(self.ellipticity[1]),
                            "center_x": center_source[0],
                            "center_y": center_source[1],
                        }]
        elif sersic_profile_str == "double":
            # w0 and w1 are the weight of the n=1 and n=4 sersic component.
            if ("w0" not in self.source_dict.colnames or 
                "w1" not in self.source_dict.colnames):
                w0 = 0.5
                w1 = 0.5
            else:
                w0 = self.source_dict["w0"]
                w1 = self.source_dict["w1"]
            size_source_arcsec0 = float(
                self.source_dict["angular_size0"]) / constants.arcsec
            size_source_arcsec1 = float(
                self.source_dict["angular_size1"]) / constants.arcsec
            ellipticity0_1 = self.source_dict["e0_1"]
            ellipticity0_2 = self.source_dict["e0_2"]
            ellipticity1_1 = self.source_dict["e1_1"]
            ellipticity1_2 = self.source_dict["e1_2"]
            kwargs_extended_source = [
                        {
                            "magnitude": w0*mag_source,
                            "R_sersic": size_source_arcsec0,
                            "n_sersic": float(self.source_dict["n_sersic_0"]),
                            "e1": float(ellipticity0_1),
                            "e2": float(ellipticity0_2),
                            "center_x": center_source[0],
                            "center_y": center_source[1],
                        }, {
                            "magnitude": w1*mag_source,
                            "R_sersic": size_source_arcsec1,
                            "n_sersic": float(self.source_dict["n_sersic_1"]),
                            "e1": float(ellipticity1_1),
                            "e2": float(ellipticity1_2),
                            "center_x": center_source[0],
                            "center_y": center_source[1],
                        }]
        else:
            raise ValueError("Provided sersic profile is not supported.")
        return kwargs_extended_source
