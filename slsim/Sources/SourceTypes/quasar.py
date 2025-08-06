from astropy.table import Column, Table
from slsim.Sources.SourceVariability.variability import (
    Variability,
    reprocess_with_lamppost_model,
)
from slsim.Sources.SourceTypes import agn
from slsim.Sources.SourceTypes.source_base import SourceBase


class Quasar(SourceBase):
    """A class to manage a quasar."""

    def __init__(self, source_dict, cosmo=None, **kwargs):
        """
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
         This dict or table should contain atleast redshift and i-band magnitude.
         eg: {"z": 0.8, "ps_mag_i": 22}
        :type source_dict: dict or astropy.table.Table
        :param cosmo: astropy.cosmology instance
        :param kwargs: dictionary of keyword arguments for a supernova. It sould contain
          following keywords:
            :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
            :type variability_model: str
            :param kwargs_variability: Keyword arguments for variability class.
            This is associated with an input for Variability class. By using these key
            words, code search for quantities in source_dict with these names and creates
            a dictionary and this dict should be passed to the Variability class.
            :type kwargs_variability: list of str
            :param lightcurve_time: observation time array for lightcurve in unit of days.
            :type lightcurve_time: array
        """

        super().__init__(source_dict=source_dict)
        self.cosmo = cosmo
        self.variability_model = kwargs.get("variability_model")
        self.kwargs_variability = kwargs.get("kwargs_variability")
        self.lightcurve_time = kwargs.get("lightcurve_time")
        self.agn_known_band = kwargs.get("agn_known_band")
        self.agn_known_mag = kwargs.get("agn_known_mag")
        self.agn_driving_variability_model = kwargs.get("agn_driving_variability_model")
        self.agn_driving_kwargs_variability = kwargs.get(
            "agn_driving_kwargs_variability"
        )
        self.input_agn_bounds_dict = kwargs.get("input_agn_bounds_dict")
        self.random_seed = kwargs.get("random_seed")

    @property
    def light_curve(self):
        if self.kwargs_variability is not None:
            kwargs_variab_extracted = {}
            z = self.source_dict["z"]
            if self.cosmo is None:
                raise ValueError(
                    "Cosmology cannot be None for AGN class. Please"
                    "provide a suitable astropy cosmology."
                )

            else:
                # Pull the agn kwarg dict out of the kwargs_variability dict
                agn_kwarg_dict = extract_agn_kwargs_from_source_dict(self.source_dict)

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
                    random_seed=self.random_seed,
                    input_agn_bounds_dict=self.input_agn_bounds_dict,
                    **agn_kwarg_dict,
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
        else:
            kwargs_variab_extracted = None
        return kwargs_variab_extracted

    def point_source_magnitude(self, band, image_observation_times=None):
        """Get the magnitude of the point source in a specific band.

        :param band: Imaging band
        :type band: str
        :param image_observation_times: Images observation time for an
            image.
        :return: Magnitude of the point source in the specified band
        :rtype: float
        """

        if not hasattr(self, "kwargs_variab_dict"):
            self.kwargs_variab_dict = self.light_curve
        column_names = self.source_dict.colnames
        if "ps_mag_" + band not in column_names:
            raise ValueError("required parameter is missing in the source dictionary.")
        else:
            band_string = "ps_mag_" + band
        if self.kwargs_variab_dict is not None:
            kwargs_variab_band = self.kwargs_variab_dict[band]
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
            return source_mag


def add_mean_mag_to_source_table(sourcedict, mean_mags, band_list):
    """This function adds/replace given mean magnitudes in given bands in a
    given source table/dict.

    :param sourcedict: Given source table.
    :param mean_mags: list of mean magnitudes in different bands.
    :param band_list: list of bands corresponding to mean_mags.
    :return: source table with additional columns corresponding to given
        mean magnitudes.
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


def extract_agn_kwargs_from_source_dict(source_dict):
    """This extracts all AGN related parameters from a source_dict Table and
    constructs a compact dictionary from them to pass into the agn class.

    :param source_dict: Astropy Table with columns representing all
        information of the source.
    :return: Compact dict object containing key+value pairs of AGN
        parameters.
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
            agn_kwarg_dict[kwarg] = source_dict[kwarg]  # .data[0]
    return agn_kwarg_dict
