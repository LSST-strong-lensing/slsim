import numpy as np
from astropy.table import Table
import random

"""References:
Sullivan et al. 2006
"""


class SupernovaeHostMatch:
    """Class to generate a host galaxy catalog for a given supernovae catalog."""

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
        :return type: dict
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

        # Iterate through the redshifts in the SNe catalog.
        for redshift in self.supernovae_catalog:

            # Select host galaxy candidates in the specified redshift range.
            # +/- 0.1 range ensures reasonable host selection size (depending on sky area).
            host_galaxy_candidates = self.galaxy_catalog[
                (self.galaxy_catalog["z"] >= (redshift - 0.1))
                & (self.galaxy_catalog["z"] <= redshift + 0.1)
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

        return matched_catalog
