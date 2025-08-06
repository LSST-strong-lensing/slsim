from astropy import cosmology
from slsim.Sources.SourceVariability.variability import Variability
from numpy import random
import numpy as np


class Agn(object):

    def __init__(
        self,
        agn_known_band,
        agn_known_mag,
        redshift,
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        lightcurve_time=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
        **kwargs_agn_model
    ):
        """Initialization of an agn.

        :param known_band: The speclite filter associated with the
            known_mag
        :param known_mag: The flux of the accretion disk in the known
            filter
        :param redshift: Redshift of the AGN
        :param cosmo: Astropy cosmology to use in calculating distances
        :param lightcurve_time: array of times associated with
            observation times
        :param agn_driving_variability_model: variability model used as
            a driving signal, This signal is then reprocessed through
            the lamppost model to get correlated signals.
        :param agn_driving_kwargs_variability: dictionary holding all
            variability keys and values
        :param kwargs_agn_model: Dictionary containing all keywords for
            the accretion disk variability model. These are:
            'black_hole_mass_exponent': mass exponent of the SMBH
            'black_hole_spin': spin of the SMBH 'inclination_angle':
            inclination of the AGN disk in degrees 'r_out': Maximum
            radius of the disk in gravitational radii 'r_resoultion':
            Number of pixels the disk is resolved to 'eddington_ratio':
            fraction of the eddington luminosity the disk is radiating
            with 'accretion_disk': accretion disk model
        """

        self.agn_known_band = agn_known_band
        self.agn_known_mag = agn_known_mag
        self.redshift = redshift
        self.kwargs_model = kwargs_agn_model
        self.cosmo = cosmo
        self.agn_driving_variability_model = agn_driving_variability_model
        self.agn_driving_kwargs_variability = agn_driving_kwargs_variability

        # Check the accretion disk type is supported
        supported_accretion_disks = ["thin_disk"]

        if self.kwargs_model["accretion_disk"] not in supported_accretion_disks:
            raise ValueError(
                "Given accretion disk model is not supported. \n Currently supported model(s) is(are):",
                supported_accretion_disks,
            )

        # Check all required parameters are defined
        required_thin_disk_params = [
            "black_hole_mass_exponent",
            "black_hole_spin",
            "inclination_angle",
            "r_out",
            "eddington_ratio",
        ]
        for kwarg in required_thin_disk_params:
            if kwarg not in self.kwargs_model.keys():
                raise ValueError("AGN parameters are not completely defined.")

        # Create the driving variability light curve from driving variability kwargs
        # The type of Variability must be a light_curve object (e.g. constructed using
        # "sinusoidal", "light_curve", "bending_power_law", or "user_defined_psd".
        driving_variability = Variability(
            self.agn_driving_variability_model, **self.agn_driving_kwargs_variability
        )

        # For consistency, we want to define the driving light curve here so it remains
        # the same for any light curve in a different filter (we don't want to
        # generate a new random light curve each time) so they remain correlated.
        # To do so, we must assure kwargs "time_array" and "magnitude_array" exist
        # in self.kwargs_model with sufficient cadence for convolution with many
        # transfer function kernels.
        if lightcurve_time is not None:
            self.kwargs_model["time_array"] = lightcurve_time

        else:
            raise ValueError(
                "Please provide an array of times to calculate the light curve at"
            )

        # Store "time_array" and "magnitude_array" kwargs which will be used to drive
        # the reprocessed signals
        self.kwargs_model["magnitude_array"] = driving_variability.variability_at_time(
            self.kwargs_model["time_array"]
        )
        self.kwargs_model["redshift"] = self.redshift

        # Create the lamppost reprocessor with a driving light curve that remains static
        self.variable_disk = Variability("lamppost_reprocessed", **self.kwargs_model)

    def get_mean_mags(self, bands):
        """Method to get mean magnitudes for AGN in multiple filters. Creates
        an accretion disk using the AccretionDiskReprocessing class in order to
        integrate the surface flux density over the accretion disk.

        :param bands: list of speclite filter names.
        :return: list of magnitudes based on the speclite bands given.
        """

        magnitudes = []
        for band in bands:
            magnitudes.append(
                self.variable_disk.accretion_disk_reprocessor.determine_agn_luminosity_from_known_luminosity(
                    self.agn_known_band,
                    self.agn_known_mag,
                    self.redshift,
                    mag_zero_point=0,
                    cosmo=self.cosmo,
                    band=band,
                )
            )
        return magnitudes


# This dictionary is designed to set the boundaries to draw random parameters from.
# The bounds of any keys may be redefined using an "input_agn_bounds_dict".
# :key black_hole_mass_exponent_bounds: mass of SMBH as log_(10)(M_{BH}/M_{sun})
# :key black_hole_spin_bounds: dimensionless spin of the SMBH. 0 is the Scwarzschild
#   case, while +/- 1 are the maximally spinning Kerr case. Negative values correspond
#   to an accretion disk that has retrograde angular momentum w.r.t. the SMBH spin vector.
# :key inclination_angle_bounds: inclination of the accretion disk w.r.t. observer.
# :key r_out_bounds: maximum radius of the SMBH in gravitational radii.
# :key eddington_ratio_bounds: a proxy for the accretion rate, defined as a fraction of
#   bolometric luminosity w.r.t. the theoretical Bondi limit. Note thin disk solutions
#   are only reasonable for relatively low accretion rates, and other disk models should
#   be used for high eddingtion ratios.
# :key supported_disk_models: List of supported disks. As the code developes, this can
#   be extended.
# :key driving_variability: List of variability objects used to \emph{drive} the intrinsic
#   light curves across all bands. The simplest case os to provide a list of variable
#   light curves directly, but this will also work with other variability choices using
#   Source.SourceVariability.variability.Variability(variability_model)
# :key intrinsic_light_curve: List of light curve objects to randomly choose from. If
#   None, then a bending power law signal will randomly be generated for a 1000 day period.
agn_bounds_dict = {
    "black_hole_mass_exponent_bounds": (6.0, 10.0),
    "black_hole_spin_bounds": (-0.997, 0.997),
    "inclination_angle_bounds": (0, 90),
    "r_out_bounds": (1000, 1000),
    "eddington_ratio_bounds": (0.01, 0.3),
    "supported_disk_models": ["thin_disk"],
    "driving_variability": ["light_curve"],
    "intrinsic_light_curve": None,
}


def RandomAgn(
    known_band,
    known_mag,
    redshift,
    cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
    lightcurve_time=None,
    agn_driving_variability_model=None,
    agn_driving_kwargs_variability=None,
    random_seed=None,
    input_agn_bounds_dict=None,
    **kwargs_agn_model
):
    """Generate a random agn.

    :param known_band: speclite filter string defining the known band
    :param known_mag: magnitude of the AGN in a known band.
    :param redshift: redshift of the AGN
    :param cosmo: Astropy cosmology to use
    :param kwargs_agn_model: Dictionary containing any fixed agn
        parameters. This will populate random agn parameters for
        keywords not given.
    """
    if random_seed is not None:
        random.seed(random_seed)

    required_agn_kwargs = [
        "black_hole_mass_exponent",
        "black_hole_spin",
        "inclination_angle",
        "r_out",
        "eddington_ratio",
    ]

    # Check if any updated bounds are input
    if input_agn_bounds_dict is not None:
        for key in agn_bounds_dict.keys():
            if key not in input_agn_bounds_dict.keys():
                input_agn_bounds_dict[key] = agn_bounds_dict[key]
    else:
        input_agn_bounds_dict = agn_bounds_dict

    # Populate any required kwargs with random values from the ranges
    # defined in the input_agn_bounds dict
    for kwarg in required_agn_kwargs:
        if kwarg not in kwargs_agn_model.keys():
            kwargs_agn_model[kwarg] = random.uniform(
                low=input_agn_bounds_dict[kwarg + "_bounds"][0],
                high=input_agn_bounds_dict[kwarg + "_bounds"][1],
            )

    # Populate keyword so a KeyError doesn't get raised when checking for
    # valid disk models
    if "accretion_disk" not in kwargs_agn_model.keys():
        kwargs_agn_model["accretion_disk"] = None

    # Check for valid disk model, otherwise populate from supported models
    # Only thin disk is supported now, but other models can be put in here once supported
    if (
        kwargs_agn_model["accretion_disk"]
        not in agn_bounds_dict["supported_disk_models"]
    ):
        random_disk_type = random.uniform(
            low=0, high=len(agn_bounds_dict["supported_disk_models"])
        )
        kwargs_agn_model["accretion_disk"] = agn_bounds_dict["supported_disk_models"][
            int(random_disk_type)
        ]

    # Check if there was a provided variabilty model.
    # If not, populate driving variability with a simple model
    if agn_driving_variability_model is None:

        # Check if other driving variabilities were inserted into bounds dict and randomize
        random_variability_type = random.uniform(
            low=0, high=len(agn_bounds_dict["driving_variability"])
        )
        kwargs_agn_model["driving_variability"] = agn_bounds_dict[
            "driving_variability"
        ][int(random_variability_type)]

        # Check if a list of other light curves were inserted into bounds dict and randomize
        if input_agn_bounds_dict["intrinsic_light_curve"] is not None:

            random_light_curve_index = random.uniform(
                low=0, high=len(input_agn_bounds_dict["intrinsic_light_curve"])
            )
            random_light_curve = input_agn_bounds_dict["intrinsic_light_curve"][
                int(random_light_curve_index)
            ]

            # Set magnitude
            random_light_curve["ps_mag_intrinsic"] += known_mag

            # Define the driving variability model as the light curve to pass into AGN object
            agn_driving_variability_model = "light_curve"
            agn_driving_kwargs_variability = random_light_curve

        # If not, generate a bending power law signal from reasonable parameters
        else:
            if lightcurve_time is None:
                length_of_required_light_curve = 1000
                lightcurve_time = np.linspace(
                    0,
                    length_of_required_light_curve - 1,
                    length_of_required_light_curve,
                )

            length_of_required_light_curve = np.max(lightcurve_time) - np.min(
                lightcurve_time
            )

            low_freq_slope = random.uniform(0, 2.0)
            random_driving_signal_kwargs = {
                "length_of_light_curve": length_of_required_light_curve,
                "time_resolution": 1,
                "log_breakpoint_frequency": random.uniform(-0.5, -2.5),
                "low_frequency_slope": low_freq_slope,
                "high_frequency_slope": random.uniform(low_freq_slope, 4.0),
                "standard_deviation": random.uniform(0.1, 1.0),
            }
            agn_driving_variability_model = "bending_power_law"
            agn_driving_kwargs_variability = random_driving_signal_kwargs

    # Define initial speclite filter to be known band
    kwargs_agn_model["speclite_filter"] = known_band

    # Generate the agn object and return it
    new_agn = Agn(
        known_band,
        known_mag,
        redshift,
        cosmo=cosmo,
        lightcurve_time=lightcurve_time,
        agn_driving_variability_model=agn_driving_variability_model,
        agn_driving_kwargs_variability=agn_driving_kwargs_variability,
        **kwargs_agn_model,
    )
    return new_agn
