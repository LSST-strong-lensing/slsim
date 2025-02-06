import numpy as np
from astropy.table import Table
import random

"""References:
Sullivan et al. 2006
"""


class SupernovaeHostMatch:
    """Class to generate a host galaxy catalog for a given supernovae
    catalog."""

    def __init__(
        self,
        supernovae_catalog,
        galaxy_catalog,
    ):
        """

        :param supernovae_catalog: supernovae redshift catalog
        :type supernovae_catalog: np.ndarray
        :param galaxy_catalog: supernovae host galaxy candidate catalog
        :type galaxy_catalog: astropy Table
        """
        self.supernovae_catalog = supernovae_catalog
        self.galaxy_catalog = galaxy_catalog

    def match(self):
        """Generates catalog in which supernovae are matched with host galaxies. (Fig 8
        - Sullivan et al. 2006)

        :return: catalog with supernovae redshifts and their corresponding host galaxies
        :return type: astropy Table
        """
        matched_catalog = Table(
            names=(
                "z",
                "M",
                "coeff",
                "ellipticity",
                "physical_size",
                "stellar_mass",
                "angular_size",
                "mag_g",
                "mag_r",
                "mag_i",
                "mag_z",
                "mag_Y",
            ),
            dtype=(
                "float64",
                "float64",
                "object",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
                "float64",
            ),
        )
        # Specify appropriate redshift range based on galaxy catalog sky area (1 deg^2 ~ 1e6
        # galaxies).
        if len(self.galaxy_catalog) > 1e6:
            range = 0.05 / 2
        else:
            range = 0.1 / 2

        # Iterate through the redshifts in the SNe catalog.
        for redshift in self.supernovae_catalog:

            # Select host galaxy candidates in the specified redshift range.
            host_galaxy_candidates = self.galaxy_catalog[
                (self.galaxy_catalog["z"] >= (redshift - range))
                & (self.galaxy_catalog["z"] <= redshift + range)
            ]

            # Calculate the weights based on stellar mass.
            log_stellar_mass_weights = 10 ** (
                (np.log10(host_galaxy_candidates["stellar_mass"])) * 0.74
            )

            # Select the host candidate based on weighting and convert to an astropy Table.
            host_galaxy = random.choices(
                host_galaxy_candidates, weights=log_stellar_mass_weights, k=1
            )

            # Append host galaxy to the matched catalog.
            matched_catalog.add_row(host_galaxy[0])
        matched_catalog["z"] = self.supernovae_catalog

        return matched_catalog
