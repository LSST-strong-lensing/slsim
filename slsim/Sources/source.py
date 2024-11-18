from slsim.Sources.SourceVariability.variability import (
    Variability,
    reprocess_with_lamppost_model,
)
import numpy as np

# from slsim.Sources.simple_supernova_lightcurve import SimpleSupernovaLightCurve
from astropy.table import Column, Table
from slsim.Sources import (
    random_supernovae,
    agn,
)
from astropy.cosmology import FLRW
from slsim.Sources import random_supernovae
from slsim.Util.param_util import ellipticity_slsim_to_lenstronomy
from slsim.Util.cosmo_util import z_scale_factor


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
        agn_known_band=None,
        agn_known_mag=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        source_type="extended",
        light_profile="single_sersic",
    ):
        """
        :param source_dict: Source properties
        :type source_dict: dict or astropy table
        :type source_dict: dict  # TODO: write what arguments need to be inculded when the source_type is interpolated.
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
        :type light_profile: str . Either "single_sersic" or "double_sersic" .
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

    @property
    def kwargs_variability_extracted(self):
        if self.kwargs_variability is not None:
            ##Here we prepare variability class on the basis of given
            # kwargs_variability.
            kwargs_variab_extracted = {}
            kwargs_variability_list = ["supernovae_lightcurve"]
            kwargs_variability_list_agn = ["agn_lightcurve"]

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
                            time=times,
                            band=provided_band,
                            zpsys=self.sn_absolute_zpsys,
                        )
                        new_column = Column([float(min(magnitudes))], name=name)
                        self._source_dict = Table(self.source_dict)
                        self._source_dict.add_column(new_column)
                        self.source_dict = self._source_dict[0]
                        kwargs_variab_extracted[element] = {
                            "MJD": times,
                            name: magnitudes,
                        }

            # Check if AGN model is used
            elif any(
                element in kwargs_variability_list_agn
                for element in list(self.kwargs_variability)
            ):

                z = self.source_dict["z"]
                if self.cosmo is None:
                    raise ValueError(
                        "Cosmology cannot be None for AGN class. Please"
                        "provide a suitable astropy cosmology."
                    )

                else:
                    # Pull the agn kwarg dict out of the kwargs_variability dict
                    agn_kwarg_dict = extract_agn_kwargs_from_source_dict(
                        self.source_dict
                    )

                    # Populate "None" for optional keys related to drawing random AGN
                    if "random_seed" in self.source_dict.colnames:
                        random_seed = self.source_dict["random_seed"][0]
                    else:
                        random_seed = None
                    if "input_agn_bounds_dict" in self.source_dict.colnames:
                        input_agn_bounds_dict = self.source_dict[
                            "input_agn_bounds_dict"
                        ][0]
                    else:
                        input_agn_bounds_dict = None

                    # If no other band and magnitude is given, populate with
                    # the assumed point source magnitude column
                    if self.agn_known_band is None:
                        if "ps_mag_i" in self.source_dict.colnames:
                            self.agn_known_band = "lsst2016-i"
                            self.agn_known_mag = self.source_dict["ps_mag_i"]
                        else:
                            raise ValueError(
                                "Please provide a band and magnitude for the AGN"
                            )

                    # Create the agn object
                    self.agn_class = agn.RandomAgn(
                        self.agn_known_band,
                        self.agn_known_mag,
                        z,
                        cosmo=self.cosmo,
                        lightcurve_time=self.lightcurve_time,
                        agn_driving_variability_model=self.agn_driving_variability_model,
                        agn_driving_kwargs_variability=self.agn_driving_kwargs_variability,
                        random_seed=random_seed,
                        input_agn_bounds_dict=input_agn_bounds_dict,
                        **agn_kwarg_dict
                    )
                    # Get mean mags for each provided band
                    # determine which kwargs_variability are lsst bands
                    lsst_bands = ["u", "g", "r", "i", "z", "y"]
                    provided_lsst_bands = set(lsst_bands) & set(self.kwargs_variability)

                    # The set "provided_lsst_bands" is no longer ordered.
                    # Therefore, create a list of speclite names in the new order
                    speclite_names = []

                    # change name to be compatible with speclite filter names
                    for band in provided_lsst_bands:
                        speclite_names.append("lsst2016-" + band)

                    # determine mean magnitudes for each band
                    mean_magnitudes = self.agn_class.get_mean_mags(speclite_names)

                    # Our input quasar catalog has magnitude only in i band. So, Agn
                    # class has computed mean magnitude of the given quasar in all lsst
                    # bands using available i-band magnitude. We want to save mean
                    # magnitudes of quasar at all bands so that we can access them at
                    # anytime.
                    self.source_dict = add_mean_mag_to_source_table(
                        self.source_dict, mean_magnitudes, provided_lsst_bands
                    )

                    # Calculate light curve in each band
                    for index, band in enumerate(provided_lsst_bands):

                        # Define name for point source mags
                        filter_name = "ps_mag_" + band

                        # Set the filter to use
                        self.agn_class.variable_disk.reprocessing_kwargs[
                            "speclite_filter"
                        ] = speclite_names[index]

                        # Set the mean magnitude of this filter
                        self.agn_class.variable_disk.reprocessing_kwargs[
                            "mean_magnitude"
                        ] = mean_magnitudes[index]

                        # Extract the reprocessed light curve
                        reprocessed_lightcurve = reprocess_with_lamppost_model(
                            self.agn_class.variable_disk
                        )

                        # Prepare the light curve to be extracted
                        times = reprocessed_lightcurve["MJD"]
                        magnitudes = reprocessed_lightcurve[
                            "ps_mag_" + speclite_names[index]
                        ]
                        # Extracts the variable light curve for each band
                        kwargs_variab_extracted[band] = {
                            "MJD": times,
                            filter_name: magnitudes,
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
            if self.kwargs_variability is not None:
                if "agn_lightcurve" not in self.kwargs_variability:
                    raise ValueError(
                        "required parameter is missing in the source dictionary."
                    )
            else:
                raise ValueError(
                    "required parameter is missing in the source dictionary."
                )
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

    def kwargs_extended_source_light(self, center_lens, draw_area, band=None):
        """Provides dictionary of keywords for the source light model(s). Kewords used
    def kwargs_extended_source_light(
        self, center_lens, draw_area, band=None, light_profile_str="single_sersic"
    ): #MODIFY THIS
        """Provids dictionary of keywords for the source light model(s). Kewords used
        are in lenstronomy conventions.

        :param center_lens: center of the deflector.
         Eg: np.array([center_x_lens, center_y_lens])
        :param draw_area: The area of the test region from which we randomly draw a
         source position. Eg: 4*pi.
        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position(
            center_lens=center_lens, draw_area=draw_area
        )
        if self.light_profile == "single_sersic":
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
        elif self.light_profile == "double_sersic":
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
        elif light_profile_str == "interpolated":
            z_image = self.source_dict["z_data"]
            pixel_width = self.source_dict["pixel_width_data"]
            pixel_width *= z_scale_factor(z_old=z_image, z_new=self.redshift)
    
            # Ensure the image is 2D
            image = self.source_dict["image"]
            if image.ndim > 2:
                image = image.squeeze()  # Remove any singleton dimensions
            if image.ndim != 2:
                raise ValueError(f"Interpolated image must be 2D, got shape {image.shape}")
    
            kwargs_extended_source = [dict(
                magnitude=mag_source,
                image=image,  # Use the potentially reshaped image
                center_x=center_source[0],
                center_y=center_source[1],
                phi_G=self.source_dict["phi_G"],
                scale=pixel_width
            )]
            print(f"kwargs_extended_source: {kwargs_extended_source}")
        else:
            raise ValueError("Provided sersic profile is not supported.")
        return kwargs_extended_source

    def extended_source_light_model(self):
        """Provides a list of source models.
        
        :return: list of extented source model.
        """
        if (
            self.source_type == "extended"
            or self.source_type == "point_plus_extended"
        ):
            if self.light_profile == "single_sersic":
                source_models_list = ["SERSIC_ELLIPSE"]
            elif self.light_profile == "double_sersic":
                source_models_list = [
                    "SERSIC_ELLIPSE",
                    "SERSIC_ELLIPSE",
                ]
            else:
                raise ValueError("Provided sersic profile is not supported. "
                            "Supported profiles are single_sersic and double_sersic.")
        else:
            source_models_list = None
        return source_models_list

def extract_agn_kwargs_from_source_dict(source_dict):
    """This extracts all AGN related parameters from a source_dict Table and constructs
    a compact dictionary from them to pass into the agn class.

    :param source_dict: Astropy Table with columns representing all information of the
        source.
    :return: Compact dict object containing key+value pairs of AGN parameters.
    """

    kwargs_variable_agn = [
        "r_out",
        "r_resolution",
        "corona_height",
        "inclination_angle",
        "black_hole_mass_exponent",
        "black_hole_spin",
        "intrinsic_light_curve",
        "eddington_ratio",
        "driving_variability_model",
        "accretion_disk",
    ]
    column_names = source_dict.colnames
    agn_kwarg_dict = {}
    for kwarg in kwargs_variable_agn:
        if kwarg in column_names:
            agn_kwarg_dict[kwarg] = source_dict[kwarg].data[0]
    return agn_kwarg_dict


def add_mean_mag_to_source_table(sourcedict, mean_mags, band_list):
    """This function adds/replace given mean magnitudes in given bands in a given source
    table/dict.

    :param sourcedict: Given source table.
    :param mean_mags: list of mean magnitudes in different bands.
    :param band_list: list of bands corresponding to mean_mags.
    :return: source table with additional columns corresponding to given mean
        magnitudes.
    """
    _source_dict = Table(sourcedict)
    for i in range(len(mean_mags)):
        new_agn_column = Column(
            [mean_mags[i]],
            name="ps_mag_" + list(band_list)[i],
        )
        if "ps_mag_" + list(band_list)[i] in _source_dict.colnames:
            _source_dict.replace_column("ps_mag_" + list(band_list)[i], new_agn_column)
        else:
            _source_dict.add_column(new_agn_column)
    return _source_dict[0]
