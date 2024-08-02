from astropy import cosmology
from slsim.Sources.SourceVariability.accretion_disk_reprocessing import (
    AccretionDiskReprocessing,
)
from speclite.filters import load_filters
from numpy import random


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

        if self.kwargs_model["accretion_disk"] not in supported_accretion_disks:
            raise ValueError(
                "Given accretion disk model is not supported. \n Currently supported model(s) is(are):",
                supported_accretion_disks,
            )

        thin_disk_params = [
            "black_hole_mass_exponent",
            "black_hole_spin",
            "inclination_angle",
            "r_out",
            "eddington_ratio",
        ]
        for kwarg in thin_disk_params:
            if kwarg not in self.kwargs_model:
                raise ValueError("AGN parameters are not completely defined.")

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

        accretion_disk = AccretionDiskReprocessing("lamppost", **self.kwargs_model)
        magnitudes = []
        for band in filters:
            magnitudes.append(
                accretion_disk.determine_agn_luminosity_from_i_band_luminosity(
                    self.i_band_mag,
                    self.redshift,
                    mag_zero_point=0,
                    cosmo=self.cosmo,
                    band=band,
                )
            )
        return magnitudes


# Include functionality to change the bounds selected from
agn_bounds_dict = {
    "black_hole_mass_exponent_bounds": [6.0, 11.0],
    "black_hole_spin_bounds": [-0.999, 0.999],
    "inclination_angle_bounds": [0, 90],
    "r_out_bounds": [1000, 1000],
    "eddington_ratio_bounds": [0.01, 0.3],
    "supported_disk_models": ["thin_disk"],
}


def RandomAgn(
    i_band_mag,
    redshift,
    cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
    seed=None,
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
    if seed:
        random.seed(seed)

    required_agn_kwargs = [
        "black_hole_mass_exponent",
        "black_hole_spin",
        "inclination_angle",
        "r_out",
        "eddington_ratio",
    ]
    for kwarg in required_agn_kwargs:
        if kwarg not in kwargs_agn_model:
            kwargs_agn_model[kwarg] = random.uniform(
                low=agn_bounds_dict[kwarg + "_bounds"][0],
                high=agn_bounds_dict[kwarg + "_bounds"][1],
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

    new_agn = Agn(
        i_band_mag,
        redshift,
        cosmo=cosmo,
        **kwargs_agn_model,
    )
    return new_agn
