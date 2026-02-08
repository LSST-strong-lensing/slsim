import copy

from slsim.Sources.SourceVariability.variability import (
    reprocess_with_lamppost_model,
)
from slsim.Sources.SourceVariability import agn
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.ImageSimulation.image_quality_lenstronomy import (
    get_speclite_filternames,
    ALL_SUPPORTED_BANDS,
)


class Quasar(SourceBase):
    """A class to manage a quasar."""

    def __init__(
        self,
        lightcurve_time=None,
        agn_known_band=None,
        agn_known_mag=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        input_agn_bounds_dict=None,
        kwargs_variability=None,
        kwargs_variability_model=None,
        variability_model="light_curve",
        random_seed=None,
        cosmo=None,
        **kwargs
    ):
        """

        :param lightcurve_time: observation time array for lightcurve in unit of days.
        :type lightcurve_time: array
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
        :param kwargs_variability_model: Pre-computed variabilities for each band (default=None)
        """

        super().__init__(
            extended_source=False,
            point_source=True,
            variability_model=variability_model,
            kwargs_variability_model=kwargs_variability_model,
            cosmo=cosmo,
            **kwargs,
        )
        self.name = "QSO"
        self._lightcurve_time = lightcurve_time
        self._agn_known_band = agn_known_band
        self._agn_known_mag = agn_known_mag
        self._agn_driving_variability_model = agn_driving_variability_model
        self._agn_driving_kwargs_variability = agn_driving_kwargs_variability
        self.input_agn_bounds_dict = input_agn_bounds_dict
        self._kwargs_variability = kwargs_variability
        if kwargs_variability_model is None:
            self._variability_computed = False
        else:
            self._variability_computed = True
        if random_seed is not None:
            self._random_seed = int(random_seed)
        else:
            self._random_seed = None

    def _init_agn_class(self):
        """Initialize the AGN class for this quasar."""
        if self._cosmo is None:
            raise ValueError(
                "Cosmology cannot be None for AGN class. Please"
                "provide a suitable astropy cosmology."
            )
        else:
            # Pull the agn kwarg dict out of the kwargs_variability dict
            agn_kwarg_dict = extract_agn_kwargs_from_source_dict(self.source_dict)

            # If no other band and magnitude is given, populate with
            # the assumed point source magnitude column
            if self._agn_known_band is None:
                if "ps_mag_i" in self.source_dict:
                    self._agn_known_band = "lsst2023-i"
                    self._agn_known_mag = self.source_dict["ps_mag_i"]
                else:
                    raise ValueError("Please provide a band and magnitude for the AGN")

            # Create the agn object
            self.agn_class = agn.RandomAgn(
                self._agn_known_band,
                self._agn_known_mag,
                self._z,
                cosmo=self._cosmo,
                lightcurve_time=self._lightcurve_time,
                agn_driving_variability_model=self._agn_driving_variability_model,
                agn_driving_kwargs_variability=self._agn_driving_kwargs_variability,
                random_seed=self._random_seed,
                input_agn_bounds_dict=self.input_agn_bounds_dict,
                **agn_kwarg_dict,
            )

    @property
    def light_curve(self):
        kwargs_variab_extracted = {}
        if self._kwargs_variability is not None:
            if not hasattr(self, "agn_class"):
                self._init_agn_class()

            # Get mean mags for each provided band if not already present in source_dict
            # determine which kwargs_variability are LSST, Roman or Euclid bands
            provided_bands = set(ALL_SUPPORTED_BANDS) & set(self._kwargs_variability)
            speclite_names = get_speclite_filternames(provided_bands)

            # determine mean magnitudes for each band using the AGN class (uses SS73 disk model)
            mean_magnitudes = self.agn_class.get_mean_mags(speclite_names)

            # check if source_dict already has mean mags for these bands
            # If yes, use those instead of computed ones
            for index, band in enumerate(provided_bands):
                ps_mag_name = "ps_mag_" + band
                if ps_mag_name in self.source_dict:
                    mean_magnitudes[index] = self.source_dict[ps_mag_name]

            # Our input quasar catalog has magnitude only in i band. So, Agn
            # class has computed mean magnitude of the given quasar in all lsst
            # bands using available i-band magnitude. We want to save mean
            # magnitudes of quasar at all bands so that we can access them at
            # anytime.
            self.source_dict = add_mean_mag_to_source_table(
                self.source_dict, mean_magnitudes, provided_bands
            )

            # Calculate light curve in each band
            for index, band in enumerate(provided_bands):

                # Define name for point source mags
                filter_name = "ps_mag_" + band

                # Set the filter to use
                self.agn_class.variable_disk.reprocessing_kwargs["speclite_filter"] = (
                    speclite_names[index]
                )

                # Set the mean magnitude of this filter
                self.agn_class.variable_disk.reprocessing_kwargs["mean_magnitude"] = (
                    mean_magnitudes[index]
                )

                # Extract the reprocessed light curve
                reprocessed_lightcurve = reprocess_with_lamppost_model(
                    self.agn_class.variable_disk
                )

                # Prepare the light curve to be extracted
                times = reprocessed_lightcurve["MJD"]
                magnitudes = reprocessed_lightcurve["ps_mag_" + speclite_names[index]]
                # Extracts the variable light curve for each band
                kwargs_variab_extracted[band] = {
                    "MJD": times,
                    filter_name: magnitudes,
                }
        self._variability_computed = True
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

        # If variability has not yet been computed, compute it now
        # this also adds the mean magnitudes to the source_dict if not already present
        if image_observation_times is not None:
            if self._variability_computed is False:
                self._kwargs_variability_model = self.light_curve

        # all the returning of variable magnitudes will be handled by the Parent class
        return super().point_source_magnitude(
            band=band, image_observation_times=image_observation_times
        )

    def update_microlensing_kwargs_source_morphology(self, kwargs_source_morphology):
        """Update the kwargs_source_morphology dictionary with AGN parameters
        from the AGN class associated with this quasar.

        :param kwargs_source_morphology: Dictionary of source morphology
            parameters.
        :return: Updated dictionary of source morphology parameters.
        """
        agn_params = [
            "black_hole_mass_exponent",
            "inclination_angle",
            "black_hole_spin",
            "eddington_ratio",
            "r_out",
            "r_resolution",
        ]
        if not hasattr(self, "agn_class"):
            self._init_agn_class()
        kwargs_agn_model = self.agn_class.kwargs_model

        for param in agn_params:
            if param not in kwargs_source_morphology:
                if param in kwargs_agn_model:
                    kwargs_source_morphology[param] = kwargs_agn_model[param]
        return kwargs_source_morphology


def add_mean_mag_to_source_table(sourcedict, mean_mags, band_list):
    """This function adds/replace given mean magnitudes in given bands in a
    given source table/dict.

    :param sourcedict: Given source table.
    :param mean_mags: list of mean magnitudes in different bands.
    :param band_list: list of bands corresponding to mean_mags.
    :return: source table with additional columns corresponding to given
        mean magnitudes.
    """
    _source_dict = copy.deepcopy(sourcedict)
    for i in range(len(mean_mags)):
        name = "ps_mag_" + list(band_list)[i]
        _source_dict[name] = mean_mags[i]

    return _source_dict


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
    # column_names = source_dict.colnames
    agn_kwarg_dict = {}
    for kwarg in kwargs_variable_agn:
        if kwarg in source_dict:
            agn_kwarg_dict[kwarg] = source_dict[kwarg]  # .data[0]
    return agn_kwarg_dict
