from astropy import cosmology
from slsim.Sources.SourceVariability.variability import(
    Variability
)
from speclite.filters import load_filters
from numpy import random
from astropy.table import Column
import numpy as np


class Agn:
    """Class for initializing an agn.

    :param i_band_mag: The flux of the accretion disk in the i_band
    :param redshift: Redshift of the AGN
    :param cosmo: Astropy cosmology to use in calculating distances
    :param kwargs_agn_model: Dictionary containing all keywords for the accretion disk
        model, in particular 'black_hole_mass_exp', 'black_hole_spin',
        'inclination_angle', 'r_out', 'r_resoultion', 'eddington_ratio', and
        'accretion_disk'
    """

    def __init__(
        self,
        i_band_mag,
        redshift,
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        **kwargs_agn_model
    ):

        self.i_band_mag = i_band_mag
        self.redshift = redshift
        self.kwargs_model = kwargs_agn_model
        self.cosmo = cosmo

        supported_accretion_disks = ["thin_disk"]
        supported_driving_variability = ["intrinsic_light_curve"]

        if self.kwargs_model["accretion_disk"] not in supported_accretion_disks:
            raise ValueError(
                "Given accretion disk model is not supported. \n Currently supported model(s) is(are):",
                supported_accretion_disks,
            )

        if self.kwargs_model["driving_variability"] not in supported_driving_variability:
            raise ValueError(
                "Given driving variability is not supported. \n Currently supported model(s) is(are):",
                supported_driving_variability,
            )

        required_thin_disk_params = [
            "black_hole_mass_exponent",
            "black_hole_spin",
            "inclination_angle",
            "r_out",
            "eddington_ratio",
        ]
        
        
        for kwarg in required_thin_disk_params:
            if kwarg not in self.kwargs_model:
                raise ValueError("AGN parameters are not completely defined.")

        if "intrinsic_light_curve" in self.kwargs_model:
            self.kwargs_model["time_array"] = self.kwargs_model["intrinsic_light_curve"]["MJD"]
            self.kwargs_model["magnitude_array"] = self.kwargs_model["intrinsic_light_curve"]["ps_mag_intrinsic"]


        # Create accretion disk object to get SED 
        #self.accretion_disk = AccretionDiskReprocessing(
        #    "lamppost",
        #    **self.kwargs_model
        #)

        # Create variability object to allow reprocessing of intrinsic signal
        # the variable_disk object has an accretion disk reprocessor which we can
        # pull the SED from.
        self.variable_disk = Variability(
            "lamppost_reprocessed",
            **self.kwargs_model
        )


    def get_mean_mags(self, survey):
        """Method to get mean magnitudes for AGN in multiple filters. Creates an
        accretion disk using the AccretionDiskReprocessing class in order to integrate
        the surface flux density over the accretion disk.

        :param survey: string representing the desired survey. Currently only accepts
            'lsst', but can be easily adjusted to incorporate any other speclite survey.
        :return: list of magnitudes based on the speclite ordering of filters in a
            survey. For lsst, this is the standard ugrizy ordering.
        """

        supported_surveys = ["lsst"]

        if survey == "lsst":
            filters = load_filters("lsst2023-*").names
        if survey not in supported_surveys:
            raise ValueError(
                "Given survey not implimented. Currently supported survey(s) is/are:",
                supported_surveys,
            )

        magnitudes = []
        for band in filters:
            magnitudes.append(
                self.variable_disk.accretion_disk_reprocessor.determine_agn_luminosity_from_i_band_luminosity(
                    self.i_band_mag,
                    self.redshift,
                    mag_zero_point=0,
                    cosmo=self.cosmo,
                    band=band,
                )
            )
        return magnitudes
        
        
basic_light_curve = {
    "MJD":np.linspace(1, 100, 100),
    "ps_mag_intrinsic":np.sin(np.linspace(1, 100, 100) * np.pi / 10)
}


agn_bounds_dict = {
    "black_hole_mass_exponent_bounds": [6.0, 10.0],
    "black_hole_spin_bounds": [-0.999, 0.999],
    "inclination_angle_bounds": [0, 90],
    "r_out_bounds": [1000, 1000],
    "eddington_ratio_bounds": [0.01, 0.3],
    "supported_disk_models": ["thin_disk"],
    "driving_variability": ["intrinsic_light_curve"],
    "intrinsic_light_curve": [basic_light_curve]
}


def RandomAgn(
    i_band_mag,
    redshift,
    cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
    random_seed=None,
    input_agn_bounds_dict=None,
    **kwargs_agn_model
):
    """Generate a random agn.

    Does not do any special parameter weighting for now, but this can be implimented
    later.
    :param i_band_mag: magnitude of the AGN in the i band
    :param redshift: redshift of the AGN
    :param cosmo: Astropy cosmology to use
    :param kwargs_agn_model: Dictionary containing any fixed agn parameters. This will
        populate random agn parameters for keywords not given.
    """
    if random_seed:
        random.seed(random_seed)

    required_agn_kwargs = [
        "black_hole_mass_exponent",
        "black_hole_spin",
        "inclination_angle",
        "r_out",
        "eddington_ratio",
    ]
    
    # Assumes input_agn_bounds_dict is an astropy.Column from source object
    if input_agn_bounds_dict is not None:
        for kwarg in required_agn_kwargs:
            if kwarg not in input_agn_bounds_dict:
                input_agn_bounds_dict.data[
                    kwarg+"_bounds"
                ] = agn_bounds_dict[kwarg+"_bounds"]
    else:
        input_agn_bounds_dict = agn_bounds_dict

    
    for kwarg in required_agn_kwargs:
        if kwarg not in kwargs_agn_model:
            kwargs_agn_model[kwarg] = random.uniform(
                low=input_agn_bounds_dict[kwarg + "_bounds"][0],
                high=input_agn_bounds_dict[kwarg + "_bounds"][1],
            )
            
    if "accretion_disk" not in kwargs_agn_model.keys():
        kwargs_agn_model["accretion_disk"] = None
        
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
        
    if "driving_variability" not in kwargs_agn_model.keys():
        random_variability_type = random.uniform(
            low=0, high=len(agn_bounds_dict["driving_variability"])
        )
        kwargs_agn_model["driving_variability"] = agn_bounds_dict["driving_variability"][
            int(random_variability_type)
        ]

    if "intrinsic_light_curve" not in kwargs_agn_model.keys():
        random_light_curve = random.uniform(
            low=0, high=len(agn_bounds_dict["intrinsic_light_curve"])
        )
        kwargs_agn_model["intrinsic_light_curve"] = agn_bounds_dict["intrinsic_light_curve"][
            int(random_light_curve)
        ]
        kwargs_agn_model["intrinsic_light_curve"]["ps_mag_intrinsic"] += i_band_mag
    


        
    kwargs_agn_model["speclite_filter"] = "lsst2023-i"

    new_agn = Agn(
        i_band_mag,
        redshift,
        cosmo=cosmo,
        **kwargs_agn_model,
    )
    return new_agn
