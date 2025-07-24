#!/usr/bin/env python
from astropy.cosmology import FlatLambdaCDM
import unittest
from slsim.Deflectors.MassLightConnection.massive_deflectors_DP0p2 import (
    find_massive_ellipticals,
)
from astropy.table import Table
import os


class TestFindPotentialLenses(unittest.TestCase):

    def setUp(self):
        # Setup mock data for the test

        # path = os.getcwd()
        # module_path, _ = os.path.split(path)
        # test_file = os.path.join(module_path, "tests/TestData/test_DP0_catalog.csv")
        test_file = os.path.join(
            os.path.dirname(__file__), "../../TestData/test_DP0_catalog.csv"
        )

        self.DP0_table = Table.read(test_file, format="csv")
        self.cosmo = FlatLambdaCDM(H0=72, Om0=0.26)

    def test_find_massive_ellipticals(self):

        # Call the function with the mock data
        DP0_table_massive_ellipticals = find_massive_ellipticals(
            DP0_table=self.DP0_table
        )

        # Ensure the function returned an Astropy Table
        self.assertIsInstance(
            DP0_table_massive_ellipticals,
            Table,
            "Expected an Astropy Table as the return type.",
        )

        # Test assertions
        # Ensure that some galaxies are identified as potential lenses
        self.assertGreater(
            len(DP0_table_massive_ellipticals),
            0,
            "A few galaxies were identified as massive ellipticals.",
        )

        print("Test passed: DP0_table_massive_ellipticals returned successfully.")


if __name__ == "__main__":
    unittest.main()
