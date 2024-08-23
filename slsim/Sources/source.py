from slsim.Sources.SourceVariability.variability import (
    Variability,
)
import numpy as np

# from slsim.Sources.simple_supernova_lightcurve import SimpleSupernovaLightCurve
from astropy.table import Column, Table
from slsim.Sources import random_supernovae
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy


class Source(object):
    """This class provides source dictionary and variable magnitude of an individual
    source."""

    def __init__(
        self,
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
        """
        :param source_dict: Source properties
        :type source_dict: dict or astropy table
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

    @property
    def kwargs_variability_extracted(self):
        if self.kwargs_variability is not None:
            ##Here we prepare variability class on the basis of given
            # kwargs_variability.
            kwargs_variab_extracted = {}
            kwargs_variability_list = ["supernovae_lightcurve"]
            # With this condition we call lightcurve generator class and prepare
            # variability class.
            if any(
                element in kwargs_variability_list
                for element in list(self.kwargs_variability)
            ):
                z = self.source_dict["z"]
                if self.cosmo is None:
                    raise ValueError(
                        "Cosmology cannot be None for Supernova class. Please"
                        "provide a suitable astropy cosmology."
                    )
                else:
                    lightcurve_class = random_supernovae.RandomizedSupernova(
                        sn_type=self.sn_type,
                        redshift=z,
                        absolute_mag=None,
                        absolute_mag_band=self.sn_absolute_mag_band,
                        mag_zpsys=self.sn_absolute_zpsys,
                        cosmo=self.cosmo,
                        modeldir=self.sn_modeldir,
                    )
                for element in list(self.kwargs_variability):
                    # if lsst filter is being used
                    if element in [
                        "r",
                        "i",
                        "g",
                        "z",
                        "y",
                        "F062",
                        "F087",
                        "F106",
                        "F129",
                        "F158",
                        "F184",
                        "F146",
                        "F213",
                    ]:
                        if element in ["r", "i", "g", "z", "y"]:
                            provided_band = "lsst" + element
                        else:
                            provided_band = element
                        name = "ps_mag_" + element
                        times = self.lightcurve_time
                        magnitudes = lightcurve_class.get_apparent_magnitude(
                            time=times, band=provided_band, zpsys=self.sn_absolute_zpsys
                        )
                        new_column = Column([float(min(magnitudes))], name=name)
                        self._source_dict = Table(self.source_dict)
                        self._source_dict.add_column(new_column)
                        self.source_dict = self._source_dict[0]
                        kwargs_variab_extracted[element] = {
                            "MJD": times,
                            name: magnitudes,
                        }
            elif "MJD" in self.kwargs_variability:
                # With this condition we extract values for kwargs_variability from the
                # given source dict and prepar variability class. Here, we expect
                # lightcurve in a source catalog and kwargs_variability should contain
                # "MJD" and "ps_mag_" + band as key.
                mag_key = []
                time_key = []
                for key in self.kwargs_variability:
                    if key.startswith("ps_mag_"):
                        mag_key.append(key)
                    else:
                        time_key.append(key)
                for element in mag_key:
                    suffix = element.split("ps_mag_")[1]
                    if element in self.source_dict.colnames:
                        if (
                            isinstance(self.source_dict[element], np.ndarray)
                            and self.source_dict[element].ndim == 2
                            and self.source_dict[element].shape[0] == 1
                        ):
                            kwargs_variab_extracted[suffix] = {
                                time_key[0]: self.source_dict[time_key[0]].reshape(-1),
                                element: self.source_dict[element].reshape(-1),
                            }
                        else:
                            kwargs_variab_extracted[suffix] = {
                                time_key[0]: self.source_dict[time_key[0]],
                                element: self.source_dict[element],
                            }
                    else:
                        raise ValueError(
                            "given keyword %s is not in the source catalog." % element
                        )
            else:
                for element in self.kwargs_variability:
                    if element in self.source_dict.colnames:
                        kwargs_variab_extracted[element] = self.source_dict[element]
                    else:
                        raise ValueError(
                            "given keyword %s is not in the source catalog." % element
                        )
        else:
            # self.variability_class = None
            kwargs_variab_extracted = None
        return kwargs_variab_extracted

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
        """Returns ellipticity components of source.
        Defined as:

        .. math::
            e1 = \\frac{1-q}{1+q} * cos(2 \\phi)
            e2 = \\frac{1-q}{1+q} * sin(2 \\phi)

        with q being the minor-to-major axis ratio.
        """

        return float(self.source_dict["e1"]), float(self.source_dict["e2"])

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """
        if not hasattr(self, "kwargs_variab_dict"):
            self.kwargs_variab_dict = self.kwargs_variability_extracted
        column_names = self.source_dict.colnames
        if "ps_mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "ps_mag_" + band
        if self.kwargs_variab_dict is not None:
            if band in self.kwargs_variab_dict.keys():
                kwargs_variab_band = self.kwargs_variab_dict[band]
            else:
                kwargs_variab_band = self.kwargs_variab_dict
            self.variability_class = Variability(
                self.variability_model, **kwargs_variab_band
            )
        else:
            self.variability_class = None
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

    def kwargs_extended_source_light(
        self, center_lens, draw_area, band=None, light_profile_str="single_sersic"
    ):
        """Provides dictionary of keywords for the source light model(s). Kewords used
        are in lenstronomy conventions.

        :param band: Imaging band
        :param light_profile_str: number of light_profile
        :type light_profile_str: str . eg: "single_sersic" or "double_sersic".
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position(
            center_lens=center_lens, draw_area=draw_area
        )
        if light_profile_str == "single_sersic":
            size_source_arcsec = float(self.angular_size)
            e1_light_source_lenstronomy, e2_light_source_lenstronomy = (
                ellipticity_slsim_to_lenstronomy(
                    e1_slsim=self.ellipticity[0], e2_slsim=self.ellipticity[1]
                )
            )
            kwargs_extended_source = [
                {
                    "magnitude": mag_source,
                    "R_sersic": size_source_arcsec,
                    "n_sersic": float(self.n_sersic),
                    "e1": e1_light_source_lenstronomy,
                    "e2": e2_light_source_lenstronomy,
                    "center_x": center_source[0],
                    "center_y": center_source[1],
                }
            ]
        elif light_profile_str == "double_sersic":
            # w0 and w1 are the weight of the n=1 and n=4 sersic component.
            if "w0" in self.source_dict.colnames or "w1" in self.source_dict.colnames:
                w0 = self.source_dict["w0"]
                w1 = self.source_dict["w1"]
            else:
                raise ValueError("weight of the light profile should be provided.")
            flux = 10 ** (-mag_source / 2.5)
            mag_source0 = -2.5 * np.log10(w0 * flux)
            mag_source1 = -2.5 * np.log10(w1 * flux)
            size_source_arcsec0 = float(self.source_dict["angular_size0"])
            size_source_arcsec1 = float(self.source_dict["angular_size1"])
            e1_light_source_1_lenstronomy, e2_light_source_1_lenstronomy = (
                ellipticity_slsim_to_lenstronomy(
                    e1_slsim=float(self.source_dict["e0_1"]),
                    e2_slsim=float(self.source_dict["e0_2"]),
                )
            )
            e1_light_source_2_lenstronomy, e2_light_source_2_lenstronomy = (
                ellipticity_slsim_to_lenstronomy(
                    e1_slsim=float(self.source_dict["e1_1"]),
                    e2_slsim=float(self.source_dict["e1_2"]),
                )
            )
            kwargs_extended_source = [
                {
                    "magnitude": mag_source0,
                    "R_sersic": size_source_arcsec0,
                    "n_sersic": float(self.source_dict["n_sersic_0"]),
                    "e1": e1_light_source_1_lenstronomy,
                    "e2": e2_light_source_1_lenstronomy,
                    "center_x": center_source[0],
                    "center_y": center_source[1],
                },
                {
                    "magnitude": mag_source1,
                    "R_sersic": size_source_arcsec1,
                    "n_sersic": float(self.source_dict["n_sersic_1"]),
                    "e1": e1_light_source_2_lenstronomy,
                    "e2": e2_light_source_2_lenstronomy,
                    "center_x": center_source[0],
                    "center_y": center_source[1],
                },
            ]
        else:
            raise ValueError("Provided sersic profile is not supported.")
        return kwargs_extended_source
