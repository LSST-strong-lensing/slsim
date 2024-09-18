import unittest
from astropy.table import Table
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pathlib import Path
from slsim.Deflectors.query_galaxies_as_deflectors import find_potential_lenses

class TestFindPotentialLenses(unittest.TestCase):

    def setUp(self):
        # Setup mock data for the test

        self.foreground_table = Table(
            {
                "redshift": np.array([0.208, 0.208, 0.200, 0.208, 0.206]),
                "gmag": np.array([19.36, 19.10, 19.16, 18.80, 20.10]),
                "rmag": np.array([18.19, 18.46, 17.99, 17.80, 18.92]),
                "imag": np.array([17.66, 18.16, 17.47, 17.37, 18.41]),
                "err_g": np.array([0.0041, 0.0036, 0.0037, 0.0032, 0.0058]),
                "err_r": np.array([0.0025, 0.0029, 0.0023, 0.0021, 0.0036]),
                "err_i": np.array([0.0022, 0.0027, 0.0020, 0.0019, 0.0031]),
                "ellipticity": np.array([0.0304, 0.2401, 0.0040, 0.0064, 0.1057]),
            }
        )

        self.background_source = "galaxy"

        self.kwargs_deflector = {
            "min_z": 0.2,
            "max_z": 1.9,
            "min_shear": 0.001,
            "max_shear": 0.02,
            "min_PA_shear": 0.0,
            "max_PA_shear": 180.0,
        }

        self.kwargs_source = {
            "type": "galaxy",
            "min_z": 0.2,
            "max_z": 5.0,
            "min_mag": 23.0,
            "max_mag": 28.0,
            "boost_csect": 3,
            "ell_min": 0.1,
            "ell_max": 0.8,
            "PA_min": 0.0,
            "PA_max": 180.0,
            "R_Einst_min": 0.5,
            "R_Einst_max": 3.0,
        }

        self.cosmo = FlatLambdaCDM(H0=72, Om0=0.26)

        self.constants = {"G": 4.2994e-9, "light_speed": 299792.458}

    def test_find_potential_lenses(self):

        # Call the function with the mock data
        find_potential_lenses(
            foreground_table=self.foreground_table,
            background_source=self.background_source,
            kwargs_deflector=self.kwargs_deflector,
            kwargs_source=self.kwargs_source,
            cosmo=self.cosmo,
            constants=self.constants,
            parallel=False,
            save_source_file=False,  # Disable file saving for the test
            chunk_offset=0,
        )

        # Test assertions

        # Set the temporary directory for file saving
        sorted_file_path = Path("sorted_galaxies.csv")

        # Check if the sorted_galaxies.csv file was created
        assert sorted_file_path.exists(), "sorted_galaxies.csv was not generated."

        sorted_table = Table.read(sorted_file_path, format="csv")
        # Ensure that some galaxies are identified as potential lenses
        self.assertGreater(
            len(sorted_table), 0, "No galaxies were identified as potential lenses."
        )

        # Check the content of the file (if necessary)
        # assert sorted_file_path.read_text() == "dummy content", "File content is not as expected."

        print("Test passed: sorted_galaxies.csv was generated successfully.")


if __name__ == "__main__":
    unittest.main()
