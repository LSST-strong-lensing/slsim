import copy

from slsim.Sources.SourceVariability.variability import (
    reprocess_with_lamppost_model,
)
from slsim.Sources.SourceVariability import agn
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.ImageSimulation.image_quality_lenstronomy import (
    get_speclite_filternames,
    get_all_supported_bands,
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
        black_hole_mass_exponent=None,
        black_hole_spin=None,
        inclination_angle=None,
        r_out=None,
        r_resolution=None,
        eddington_ratio=None,
        accretion_disk=None,
        corona_height=None,
        driving_variability_model=None,
        intrinsic_light_curve=None,
        **kwargs
    ):
        """

        AGN accretion-disk parameters below (all optional). Any left as ``None`` are
        drawn randomly by :func:`~slsim.Sources.SourceVariability.agn.RandomAgn` from
        ``agn_bounds_dict`` (or ``input_agn_bounds_dict`` if provided) the first time
        the AGN model is needed (on first access of ``.light_curve`` or
        ``update_microlensing_kwargs_source_morphology``) -- this is a deliberate
        generative feature (population diversity), not a fallback to guard against.
        See :mod:`slsim.Sources.SourceVariability.agn` for the sampling distributions.

        :param lightcurve_time: time array for lightcurve in unit of days,
         defined in the rest (source) frame relative to time_zero_point (see
         Source._image_to_source_time_translation()). The AGN reprocessing response
         function is defined by the disk's physical light-crossing time, which is
         redshift-independent, so this is in rest-frame.
        :type lightcurve_time: array
        :param cosmo: astropy.cosmology instance
        :param kwargs: required per-object catalog data (dict or one row of an
         Astropy table), passed through to `SourceBase`. Must contain at least
         redshift (``z``) and i-band magnitude (``ps_mag_i``).
         eg: {"z": 0.8, "ps_mag_i": 22}
        :param variability_model: keyword for variability model to be used. This is an
            input for the Variability class.
        :type variability_model: str
        :param kwargs_variability: Keyword arguments for variability class.
            This is associated with an input for Variability class. By using these keywords,
            code search for quantities in source_dict with these names and creates
            a dictionary and this dict should be passed to the Variability class.
        :type kwargs_variability: list of str
        :param kwargs_variability_model: Pre-computed variabilities for each band (default=None)
        :param black_hole_mass_exponent: mass exponent of the SMBH,
         log10(M_BH/M_sun). If ``None``, drawn from
         ``agn_bounds_dict["black_hole_mass_exponent_bounds"]``.
        :type black_hole_mass_exponent: float or None
        :param black_hole_spin: dimensionless spin of the SMBH, in [-1, 1]. 0 is
         Schwarzschild; negative values are retrograde relative to the disk. If
         ``None``, drawn from ``agn_bounds_dict["black_hole_spin_bounds"]``.
        :type black_hole_spin: float or None
        :param inclination_angle: inclination of the AGN disk w.r.t. the observer,
         in degrees. If ``None``, drawn from
         ``agn_bounds_dict["inclination_angle_bounds"]``.
        :type inclination_angle: float or None
        :param r_out: maximum radius of the disk, in gravitational radii. If
         ``None``, drawn from ``agn_bounds_dict["r_out_bounds"]``.
        :type r_out: float or None
        :param eddington_ratio: fraction of the Eddington luminosity the disk is
         radiating with (accretion-rate proxy; thin-disk solutions are only valid
         for relatively low values). If ``None``, drawn from
         ``agn_bounds_dict["eddington_ratio_bounds"]``.
        :type eddington_ratio: float or None
        :param accretion_disk: accretion disk model name. Currently only
         ``"thin_disk"`` is supported. If ``None``, chosen randomly from
         ``agn_bounds_dict["supported_disk_models"]`` (currently only one option).
        :type accretion_disk: str or None
        :param r_resolution: number of radial pixels the disk is resolved to.
         Unlike the five parameters above, this is *not* part of `RandomAgn`'s
         random-fill set -- if left ``None``, a default is applied deeper inside
         the lamppost reprocessing model (see
         :mod:`slsim.Sources.SourceVariability.variability`).
        :type r_resolution: int or None
        :param corona_height: height of the X-ray corona above the disk, used in
         the lamppost reprocessing model. Same not-randomly-filled caveat as
         ``r_resolution``.
        :type corona_height: float or None
        :param driving_variability_model: reserved/rarely-used passthrough key
         consumed only by the lamppost reprocessing model directly; do not confuse
         with ``agn_driving_variability_model`` above, which is the parameter
         actually used to select the driving-signal model for `RandomAgn`/`Agn`.
         Leave as ``None`` unless you specifically know you need it.
        :type driving_variability_model: str or None
        :param intrinsic_light_curve: reserved/rarely-used passthrough key; the
         actual per-draw mechanism for supplying a library of intrinsic light
         curves to sample from is ``input_agn_bounds_dict["intrinsic_light_curve"]``
         (a list), not this key. Leave as ``None`` unless you specifically know you
         need it.
        :type intrinsic_light_curve: object or None
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
        self._black_hole_mass_exponent = black_hole_mass_exponent
        self._black_hole_spin = black_hole_spin
        self._inclination_angle = inclination_angle
        self._r_out = r_out
        self._r_resolution = r_resolution
        self._eddington_ratio = eddington_ratio
        self._accretion_disk = accretion_disk
        self._corona_height = corona_height
        self._driving_variability_model = driving_variability_model
        self._intrinsic_light_curve = intrinsic_light_curve
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
            # Build the AGN disk-physics kwarg dict from the explicit constructor
            # parameters, omitting any left as None so RandomAgn's random-fill
            # behavior (see agn.py) kicks in for those -- a key present with value
            # None would suppress that random-fill instead of triggering it.
            agn_kwarg_dict = {
                key: value
                for key, value in {
                    "black_hole_mass_exponent": self._black_hole_mass_exponent,
                    "black_hole_spin": self._black_hole_spin,
                    "inclination_angle": self._inclination_angle,
                    "r_out": self._r_out,
                    "r_resolution": self._r_resolution,
                    "eddington_ratio": self._eddington_ratio,
                    "accretion_disk": self._accretion_disk,
                    "corona_height": self._corona_height,
                    "driving_variability_model": self._driving_variability_model,
                    "intrinsic_light_curve": self._intrinsic_light_curve,
                }.items()
                if value is not None
            }

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
            provided_bands = set(get_all_supported_bands()) & set(
                self._kwargs_variability
            )
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
