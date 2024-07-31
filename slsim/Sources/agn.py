from astropy import cosmology
from slsim.Sources.SourceVariability.accretion_disk_reprocessing import (
    AccretionDiskReprocessing,
)
from speclite.filters import load_filters


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
            "r_resolution",
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
