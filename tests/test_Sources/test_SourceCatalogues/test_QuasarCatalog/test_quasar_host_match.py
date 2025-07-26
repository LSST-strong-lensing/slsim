import numpy as np
import numpy.testing as npt  # Import numpy testing
import pytest
from astropy.table import Table
from astropy.table import vstack
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from unittest.mock import patch
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline

# Import the functions and class to be tested
from slsim.Sources.SourceCatalogues.QuasarCatalog.quasar_host_match import (
    sample_eddington_rate,
    black_hole_mass_from_vel_disp,
    calculate_lsst_magnitude,
    QuasarHostMatch,
)


def test_black_hole_mass_from_vel_disp():
    """Tests the calculation of black hole mass from velocity dispersion."""
    # Test with a standard velocity dispersion of 200 km/s
    sigma = 200
    expected_mass = 10**9 * 0.310 * (200 / 200) ** 4.38
    # Use npt.assert_almost_equal for numerical comparison
    npt.assert_almost_equal(black_hole_mass_from_vel_disp(sigma), expected_mass)

    # Test with a different velocity dispersion
    sigma_2 = 150
    expected_mass_2 = 10**9 * 0.310 * (150 / 200) ** 4.38
    npt.assert_almost_equal(black_hole_mass_from_vel_disp(sigma_2), expected_mass_2)


def test_calculate_lsst_magnitude():
    """Tests the calculation of absolute magnitude for different LSST bands."""
    bh_mass = 1e8  # 100 million solar masses
    eddington_ratio = 0.5

    # Test a valid band ('i' band)
    mag_i = calculate_lsst_magnitude("i", bh_mass, eddington_ratio)
    l_bol = (3.2e4 * bh_mass) * eddington_ratio
    m_bol = 4.74 - 2.5 * np.log10(l_bol)
    bc_i = 8.1
    expected_mag_i = m_bol + 2.5 * np.log10(bc_i)
    npt.assert_almost_equal(mag_i, expected_mag_i, decimal=6)

    # Test that an invalid band raises a ValueError
    with pytest.raises(ValueError):
        calculate_lsst_magnitude("x", bh_mass, eddington_ratio)


def test_sample_eddington_rate():
    """Tests the sampling of Eddington ratios."""
    np.random.seed(42)

    # Test sampling a single value
    z = 1.0
    result_array = sample_eddington_rate(z, size=1)
    assert isinstance(result_array, np.ndarray)
    assert len(result_array) == 1
    lambda_val = result_array.item()  # Extract scalar value
    assert isinstance(lambda_val, float)
    assert 0.1 <= lambda_val <= 1.0
    # Use npt for a precise check against the known seeded value
    npt.assert_almost_equal(lambda_val, 0.29617, decimal=5)

    # Test sampling multiple values
    n_samples = 5
    lambda_vals = sample_eddington_rate(z, size=n_samples)
    assert isinstance(lambda_vals, np.ndarray)
    assert len(lambda_vals) == n_samples
    assert np.all((lambda_vals >= 0.1) & (lambda_vals <= 1.0))


class TestQuasarHostMatch:
    """Test suite for the QuasarHostMatch class."""

    @pytest.fixture
    def setup_catalogs(self):
        """Creates basic quasar and galaxy catalogs for testing."""
        quasar_cat = Table({"z": [0.5], "M_i": [-23.0]})

        galaxy_cat = Table(
            {
                "z": [0.5001, 0.4901],
                "vel_disp": [150.0, 200.0],  # km/s
                "stellar_mass": [1e11, 2e11],  # Solar masses
                "host_id": [1, 2],  # Unique identifier for host galaxies
            }
        )
        return quasar_cat, galaxy_cat

    def test_initialization(self, setup_catalogs):
        """Tests the initialization of the QuasarHostMatch class."""
        quasar_cat, galaxy_cat = setup_catalogs

        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(), galaxy_catalog=galaxy_cat.copy()
        )

        assert matcher.quasar_catalog is not None
        assert matcher.galaxy_catalog is not None
        assert len(matcher.quasar_catalog) == len(quasar_cat)
        assert len(matcher.galaxy_catalog) == len(galaxy_cat)

    @patch("slsim.Sources.QuasarCatalog.quasar_host_match.sample_eddington_rate")
    def test_match_successful(self, mock_sample_rate, setup_catalogs):
        """Tests a successful match between a quasar and its host galaxy."""
        quasar_cat, galaxy_cat = setup_catalogs

        # Pass a copy of the galaxy catalog to avoid modifying the fixture's state
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(), galaxy_catalog=galaxy_cat.copy()
        )

        # Mock the sample_eddington_rate function
        mock_sample_rate.return_value = np.array([0.5])  # Mocked Eddington ratio

        result_table = matcher.match()

        # Check that the result is a Table with the expected number of rows
        assert isinstance(result_table, Table)
        assert len(result_table) == 1

        # resulting table should have the expected columns: z, M_i, black_hole_mass_exponent, eddington_ratio, stellar_mass, vel_disp
        assert "z" in result_table.colnames
        assert "M_i" in result_table.colnames
        assert "black_hole_mass_exponent" in result_table.colnames
        assert "eddington_ratio" in result_table.colnames
        assert "stellar_mass" in result_table.colnames
        assert "vel_disp" in result_table.colnames

        # We pre-calculated that host_id=1 is the better match
        assert result_table["host_id"][0] == 1
        assert result_table["vel_disp"][0] == 150.0
        assert result_table["stellar_mass"][0] == 1e11
        assert "black_hole_mass_exponent" in result_table.colnames
        assert "eddington_ratio" in result_table.colnames
        assert result_table["eddington_ratio"][0] == 0.5

    def test_match_raises_error_if_no_vel_disp(self, setup_catalogs):
        """Tests that a ValueError is raised if 'vel_disp' is missing."""
        quasar_cat, galaxy_cat = setup_catalogs
        galaxy_cat_no_vel_disp = galaxy_cat.copy()
        galaxy_cat_no_vel_disp.remove_column("vel_disp")

        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat, galaxy_catalog=galaxy_cat_no_vel_disp
        )

        with pytest.raises(ValueError, match="must have 'vel_disp' column"):
            matcher.match()

    def test_no_host_candidates_in_range(self):
        """Tests that a quasar is skipped if no host galaxies are in the
        redshift range."""
        # A quasar with a redshift that is far from any galaxy in the catalog
        quasar_cat = Table({"z": [2.0], "M_i": [-23.0]})
        # A galaxy catalog with redshifts far from the quasar
        galaxy_cat = Table(
            {
                "z": [0.5001, 0.4901],
                "vel_disp": [150.0, 200.0],
                "stellar_mass": [1e11, 2e11],
                "host_id": [1, 2],
            }
        )

        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(), galaxy_catalog=galaxy_cat.copy()
        )
        result_table = matcher.match()

        # Check that the resulting table is empty because no match was found
        assert isinstance(result_table, Table)
        assert len(result_table) == 0

    def test_large_area(self):
        """Tests the matching with a large galaxy catalog."""
        quasar_cat = Table({"z": [0.5], "M_i": [-23.0], "ps_mag_i": [23.0]})

        skypy_config = None
        sky_area = Quantity(1, unit="deg2")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        pipeline = SkyPyPipeline(
            skypy_config=skypy_config,
            sky_area=sky_area,
            filters=None,
            cosmo=cosmo,
        )
        host_galaxy_catalog = vstack(
            [pipeline.red_galaxies, pipeline.blue_galaxies],
            join_type="exact",
        )
        print(f"Host galaxy catalog size: {len(host_galaxy_catalog)}")

        self._f_vel_disp = vel_disp_abundance_matching(
            host_galaxy_catalog,
            z_max=0.5,
            sky_area=sky_area,
            cosmo=cosmo,
        )
        host_galaxy_catalog["vel_disp"] = self._f_vel_disp(
            np.log10(host_galaxy_catalog["stellar_mass"])
        )

        galaxy_cat = host_galaxy_catalog.copy()

        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(), galaxy_catalog=galaxy_cat.copy()
        )

        rslt = matcher.match()

        assert isinstance(rslt, Table)
        assert len(rslt) == 1  # Expecting one match for the single quasar
