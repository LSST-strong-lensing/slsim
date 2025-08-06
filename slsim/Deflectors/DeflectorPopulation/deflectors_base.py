from abc import ABC, abstractmethod
import numpy as np


class DeflectorsBase(ABC):
    """Abstract Base Class to create a class that accesses a set of deflectors.

    All object that inherit from Lensed System must contain the methods
    it contains.
    """

    def __init__(
        self,
        deflector_table,
        kwargs_cut,
        cosmo,
        sky_area,
        gamma_pl=None,
        deflector_type="EPL_SERSIC",
    ):
        """

        :param deflector_table: table with lens parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :param deflector_type: type of Deflector() model class
        :type deflector_type: string
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation
         of a density slope for gaussian distribution or minimum and maximum values of
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        """
        self.deflector_table = deflector_table
        self.kwargs_cut = kwargs_cut
        self.cosmo = cosmo
        self.sky_area = sky_area
        self.deflector_profile = deflector_type
        galaxies = deflector_table
        galaxy_number = len(galaxies)
        if gamma_pl is not None:
            if isinstance(gamma_pl, float):
                galaxies["gamma_pl"] = [gamma_pl] * galaxy_number
            elif isinstance(gamma_pl, dict):
                parameters = gamma_pl.keys()
                if "mean" in parameters and "std_dev" in parameters:
                    slope_list = np.random.normal(
                        loc=gamma_pl["mean"],
                        scale=gamma_pl["std_dev"],
                        size=galaxy_number,
                    )
                elif "gamma_min" in parameters and "gamma_max" in parameters:
                    slope_list = np.random.uniform(
                        low=gamma_pl["gamma_min"],
                        high=gamma_pl["gamma_max"],
                        size=galaxy_number,
                    )
                else:
                    raise ValueError(
                        "The given quantities in gamma_pl are not recognized."
                        " Please provide the mean and standard deviation for a"
                        " gaussian distribution, or specify the gamma_min and gamma_max "
                        " for a uniform distribution."
                    )
                galaxies["gamma_pl"] = slope_list
            else:
                raise ValueError(
                    "The given format of the gamma_pl is not supported."
                    " Please provide a float or dictionary. See the documentation"
                    " in DeflectorsBase class"
                )

    @abstractmethod
    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        pass

    @abstractmethod
    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """
        pass
