from slsim.Sources.SourceVariability.variability import (
    Variability,
)
import numpy as np

# from slsim.Sources.simple_supernova_lightcurve import SimpleSupernovaLightCurve
from astropy.table import Column, Table
from slsim.Sources import random_supernovae


class Source(object):
    """This class provides source dictionary and variable magnitude of a individual
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
    ):
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
        :param sn_type: Supernova type (Ia, Ib, Ic, IIP, etc.)
        :type sn_type: str
        :param sn_absolute_mag_band: Band used to normalize to absolute magnitude
        :type sn_absolute_mag_band: str or `~sncosmo.Bandpass`
        :param sn_absolute_zpsys: Optional, AB or Vega (AB default)
        :type sn_absolute_zpsys: str
        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array
        """
        self.source_dict = source_dict
        if kwargs_variability is not None:
            ##Here we prepare variability class on the basis of given
            # kwargs_variability.
            kwargs_variab_extracted = {}
            kwargs_variability_list = ["supernovae_lightcurve"]
            # With this condition we call lightcure generator class and prepare
            # variability class.
            if any(
                element in kwargs_variability_list
                for element in list(kwargs_variability)
            ):
                z = self.source_dict["z"]
                if cosmo is None:
                    raise ValueError(
                        "Cosmology cannot be None for Supernova class. Please"
                        "provide a suitable astropy cosmology."
                    )
                else:
                    lightcurve_class = random_supernovae.RandomizedSupernova(
                        sn_type=sn_type,
                        redshift=z,
                        absolute_mag=None,
                        absolute_mag_band=sn_absolute_mag_band,
                        mag_zpsys=sn_absolute_zpsys,
                        cosmo=cosmo,
                    )
                for element in list(kwargs_variability):
                    # if lsst filter is being used
                    if element in ["r", "i", "g"]:
                        provided_band = "lsst" + element
                        name = "ps_mag_" + element
                    # if roman filter is being used
                    elif element in ["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"]:
                        provided_band = element
                        name = "ps_mag_"+ element
                times = lightcurve_time
                magnitudes = lightcurve_class.get_apparent_magnitude(
                    time=times, band=provided_band, zpsys=sn_absolute_zpsys
                )
                new_column = Column(
                    [float(min(magnitudes))], name=name
                )
                self._source_dict = Table(self.source_dict)
                self._source_dict.add_column(new_column)
                self.source_dict = self._source_dict[0]
                kwargs_variab_extracted["MJD"] = times
                kwargs_variab_extracted[name] = magnitudes
            else:
                # With this condition we extract values for kwargs_variability from the
                # given source dict and prepar variability class.
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
            self.variability_class = None

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

    def kwargs_extended_source_light(
        self, center_lens, draw_area, band=None, light_profile_str="single_sersic"
    ):
        """Provids dictionary of keywords for the source light model(s). Kewords used
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
            kwargs_extended_source = [
                {
                    "magnitude": mag_source,
                    "R_sersic": size_source_arcsec,
                    "n_sersic": float(self.n_sersic),
                    "e1": float(self.ellipticity[0]),
                    "e2": float(self.ellipticity[1]),
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
            ellipticity0_1 = self.source_dict["e0_1"]
            ellipticity0_2 = self.source_dict["e0_2"]
            ellipticity1_1 = self.source_dict["e1_1"]
            ellipticity1_2 = self.source_dict["e1_2"]
            kwargs_extended_source = [
                {
                    "magnitude": mag_source0,
                    "R_sersic": size_source_arcsec0,
                    "n_sersic": float(self.source_dict["n_sersic_0"]),
                    "e1": float(ellipticity0_1),
                    "e2": float(ellipticity0_2),
                    "center_x": center_source[0],
                    "center_y": center_source[1],
                },
                {
                    "magnitude": mag_source1,
                    "R_sersic": size_source_arcsec1,
                    "n_sersic": float(self.source_dict["n_sersic_1"]),
                    "e1": float(ellipticity1_1),
                    "e2": float(ellipticity1_2),
                    "center_x": center_source[0],
                    "center_y": center_source[1],
                },
            ]
        else:
            raise ValueError("Provided sersic profile is not supported.")
        return kwargs_extended_source
