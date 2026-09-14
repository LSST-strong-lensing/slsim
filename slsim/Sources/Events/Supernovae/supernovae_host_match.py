import numpy as np

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
        # Specify appropriate redshift range based on galaxy catalog sky area (1 deg^2 ~ 1e6
        # galaxies).
        if len(self.galaxy_catalog) > 1e6:
            range = 0.05 / 2
        else:
            range = 0.1 / 2

        galaxy_redshifts = np.asarray(self.galaxy_catalog["z"], dtype=float)
        redshift_order = np.argsort(galaxy_redshifts)
        sorted_redshifts = galaxy_redshifts[redshift_order]

        # Calculate the weights based on stellar mass => m ** 0.74, evaluated here for the whole catalog at once.
        weights = np.asarray(self.galaxy_catalog["stellar_mass"], dtype=float) ** 0.74

        # Iterate through the redshifts in the SNe catalog.
        matched_indices = []
        for redshift in self.supernovae_catalog:

            # Select host galaxy candidates in the specified redshift range.
            low = np.searchsorted(sorted_redshifts, redshift - range, "left")
            high = np.searchsorted(sorted_redshifts, redshift + range, "right")
            candidates = redshift_order[low:high]

            # Select the host candidate based on weighting.
            candidate_weights = weights[candidates]
            matched_indices.append(
                np.random.choice(
                    candidates, p=candidate_weights / candidate_weights.sum()
                )
            )

        # Index the catalog once, instead of growing a table row by row.
        #  maintains the original units (angular sizes coming out of the SkyPy pipeline are in arcsec.)
        matched_catalog = self.galaxy_catalog[matched_indices]
        matched_catalog["z"] = self.supernovae_catalog

        return matched_catalog
