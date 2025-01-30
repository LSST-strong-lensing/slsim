#!/usr/bin/env python
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.massive_deflectors_DP0p2 import get_galaxy_parameters_from_moments
import unittest
from slsim.Deflectors.massive_deflectors_DP0p2 import find_massive_ellipticals
from astropy.table import Table


def test_get_galaxy_parameters_from_moments():

    xx, xy, yy = 8.8278519, -2.2316406, 11.4708189

    theta, r_eff, q, ellipticity = get_galaxy_parameters_from_moments(xx, xy, yy)

    np.testing.assert_equal(len(theta), len(r_eff), len(q), len(ellipticity))
    np.testing.assert_almost_equal(theta, -119, decimal=-1)
    np.testing.assert_almost_equal(r_eff, 3.1324, decimal=0)
    np.testing.assert_almost_equal(q, 0.77, decimal=0)
    np.testing.assert_almost_equal(ellipticity, 0.2299, decimal=0)


class TestFindPotentialLenses(unittest.TestCase):

    def setUp(self):
        # Setup mock data for the test

        self.DP0_table = Table.read("../TestData/test_DP0_catalog.csv")

        self.cosmo = FlatLambdaCDM(H0=72, Om0=0.26)

        self.constants = {"G": 4.2994e-9, "light_speed": 299792.458}

    def test_find_massive_ellipticals(self):

        # Call the function with the mock data
        DP0_table_massive_ellipticals = find_massive_ellipticals(
            DP0_table=self.DP0_table
        )

        # Test assertions
        # Ensure that some galaxies are identified as potential lenses
        self.assertGreater(
            len(DP0_table_massive_ellipticals), 3, "3 galaxies were identified as massive ellipticals."
        )

        print("Test passed: DP0_table_massive_ellipticals returned successfully.")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    test_get_galaxy_parameters_from_moments()
    unittest.main()
