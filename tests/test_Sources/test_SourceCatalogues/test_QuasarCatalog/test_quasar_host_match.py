import numpy as np
import numpy.testing as npt
import pytest
from astropy.table import Table
from astropy.table import vstack
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline

from slsim.Sources.SourceCatalogues.QuasarCatalog.quasar_host_match import (
    L_EDDINGTON_PER_MSUN,
    QuasarHostMatch,
    absolute_i_magnitude_from_l3000,
    black_hole_mass_from_vel_disp,
    bolometric_luminosity_from_l3000,
    calculate_lsst_magnitude,
    eddington_ratio_grid,
    l3000_from_absolute_i_magnitude,
    sample_eddington_rate,
    SHEN11_LOG_LAMBDA_MEAN,
)


def test_black_hole_mass_from_vel_disp():
    """Tests the calculation of black hole mass from velocity dispersion."""
    npt.assert_allclose(black_hole_mass_from_vel_disp(200), 10**9 * 0.310)
    npt.assert_allclose(
        black_hole_mass_from_vel_disp(150), 10**9 * 0.310 * (150 / 200) ** 4.38
    )

    # the intrinsic scatter is lognormal and unbiased in the log
    np.random.seed(42)
    sigma = np.full(200000, 200.0)
    scattered = np.log10(black_hole_mass_from_vel_disp(sigma, scatter=0.29))
    npt.assert_allclose(scattered.mean(), np.log10(0.310e9), atol=0.01)
    npt.assert_allclose(scattered.std(), 0.29, atol=0.01)


def test_calculate_lsst_magnitude():
    """Tests the absolute AB magnitude of a quasar in an LSST band."""
    mag_i = calculate_lsst_magnitude("i", 1e8, 0.5)

    # reproduce the AB magnitude by hand from nu L_nu at the i band pivot
    l_bol = L_EDDINGTON_PER_MSUN * 1e8 * 0.5
    nu = 2.99792458e17 / 754.6
    flux = l_bol / 8.1 / nu / (4 * np.pi * 3.0856775814913673e19**2)
    npt.assert_allclose(mag_i, -2.5 * np.log10(flux) - 48.60)

    # brighter bands for a more massive black hole
    assert calculate_lsst_magnitude("i", 1e9, 0.5) < mag_i

    with pytest.raises(ValueError):
        calculate_lsst_magnitude("x", 1e8, 0.5)


def test_sample_eddington_rate():
    """Tests the sampling of Eddington ratios."""
    np.random.seed(42)

    result = sample_eddington_rate(1.0, size=1)
    assert isinstance(result, np.ndarray)
    assert len(result) == 1
    assert 0.1 <= result.item() <= 1.0

    samples = sample_eddington_rate(1.0, size=200000)
    assert np.all((samples >= 0.1) & (samples <= 1.0))

    # the sampled distribution follows the lambda**gamma_e power law
    gamma_e = -0.65
    expected_median = (
        0.5 * (1.0 ** (gamma_e + 1) - 0.1 ** (gamma_e + 1)) + 0.1 ** (gamma_e + 1)
    ) ** (1 / (gamma_e + 1))
    npt.assert_allclose(np.median(samples), expected_median, rtol=0.01)


def test_sample_eddington_rate_is_redshift_independent():
    """The redshift only enters the amplitude, which normalisation removes."""
    np.random.seed(0)
    low_z = sample_eddington_rate(0.1, size=1000)
    np.random.seed(0)
    high_z = sample_eddington_rate(4.0, size=1000)
    npt.assert_array_equal(low_z, high_z)


def test_eddington_ratio_grid():
    grid, pdf = eddington_ratio_grid(lambda_min=0.01, lambda_max=1.0, n_grid=10)
    assert len(grid) == len(pdf) == 10
    npt.assert_allclose([grid[0], grid[-1]], [0.01, 1.0])
    npt.assert_allclose(pdf, grid**-0.65)

    with pytest.raises(ValueError):
        eddington_ratio_grid(distribution="uniform")


def test_lognormal_eddington_ratio_grid():
    """The lognormal shape is a Gaussian in log10(lambda) centred on the
    requested mean, unlike the power law whose mass piles up at lambda ~ 1."""
    kwargs = dict(lambda_min=1e-3, lambda_max=3.0, n_grid=4000)

    def median(distribution):
        grid, pdf = eddington_ratio_grid(distribution=distribution, **kwargs)
        # mass per unit ln(lambda) is lambda * p(lambda) on a log grid
        cumulative = np.cumsum(pdf * grid)
        return np.log10(np.interp(0.5, cumulative / cumulative[-1], grid))

    npt.assert_allclose(median("lognormal"), SHEN11_LOG_LAMBDA_MEAN, atol=0.01)
    assert median("power_law") > median("lognormal")


def test_luminosity_conversions():
    """The magnitude to luminosity conversions invert each other and follow
    the expected 0.4 dex per magnitude scaling."""
    m_i = np.array([-28.0, -26.0, -24.0])
    l3000 = l3000_from_absolute_i_magnitude(m_i)
    npt.assert_allclose(absolute_i_magnitude_from_l3000(l3000), m_i)
    npt.assert_allclose(np.diff(np.log10(l3000)), -0.8)

    # an M_i = -26 quasar should sit near log L_bol = 46.2, as observed
    l_bol = bolometric_luminosity_from_l3000(l3000_from_absolute_i_magnitude(-26.0))
    npt.assert_allclose(np.log10(l_bol), 46.2, atol=0.15)

    linear = bolometric_luminosity_from_l3000(1e46)
    npt.assert_allclose(linear, 5.18e46)
    npt.assert_allclose(
        bolometric_luminosity_from_l3000(1e46, anisotropy_correction=0.75),
        0.75 * linear,
    )
    assert bolometric_luminosity_from_l3000(1e46, form="log") != linear
    with pytest.raises(ValueError):
        bolometric_luminosity_from_l3000(1e46, form="quadratic")


class TestQuasarHostMatch:
    """Test suite for the QuasarHostMatch class."""

    @pytest.fixture
    def setup_catalogs(self):
        """Creates basic quasar and galaxy catalogs for testing."""
        quasar_cat = Table({"z": [0.5], "M_i": [-23.0]})
        galaxy_cat = Table(
            {
                "z": [0.5001, 0.4999],
                "vel_disp": [150.0, 200.0],
                "stellar_mass": [1e11, 2e11],
                "galaxy_type": ["blue", "red"],
                "host_id": [1, 2],
            }
        )
        return quasar_cat, galaxy_cat

    def test_initialization(self, setup_catalogs):
        quasar_cat, galaxy_cat = setup_catalogs
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(), galaxy_catalog=galaxy_cat.copy()
        )
        assert len(matcher.quasar_catalog) == len(quasar_cat)
        assert len(matcher.galaxy_catalog) == len(galaxy_cat)
        assert matcher.n_rejected == 0

    def test_match_successful(self, setup_catalogs):
        quasar_cat, galaxy_cat = setup_catalogs
        np.random.seed(7)
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(),
            galaxy_catalog=galaxy_cat.copy(),
            min_candidates=2,
            progress=False,
        )
        result = matcher.match()

        assert isinstance(result, Table)
        assert len(result) == 1
        for column in [
            "z",
            "M_i",
            "black_hole_mass_exponent",
            "eddington_ratio",
            "log_bolometric_luminosity",
            "stellar_mass",
            "vel_disp",
        ]:
            assert column in result.colnames
        assert result["host_id"][0] in [1, 2]

    def test_match_is_self_consistent(self):
        """The assigned black hole mass and Eddington ratio must reproduce the
        bolometric luminosity implied by the quasar magnitude."""
        np.random.seed(3)
        n_gal = 20000
        galaxy_cat = Table(
            {
                "z": np.random.uniform(0.1, 2.0, n_gal),
                "vel_disp": 10 ** np.random.normal(2.15, 0.16, n_gal),
            }
        )
        quasar_cat = Table(
            {
                "z": np.random.uniform(0.3, 1.8, 2000),
                "M_i": np.random.uniform(-26.0, -22.0, 2000),
            }
        )
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat,
            galaxy_catalog=galaxy_cat,
            min_candidates=50,
            progress=False,
        )
        result = matcher.match()
        assert len(result) > 0.9 * len(quasar_cat)

        log_l_edd = (
            result["black_hole_mass_exponent"]
            + np.log10(L_EDDINGTON_PER_MSUN)
            + np.log10(result["eddington_ratio"])
        )
        npt.assert_allclose(
            result["log_bolometric_luminosity"], log_l_edd, atol=1e-10
        )

        # the M-sigma scatter is recovered, rather than the relation being exact
        residual = result["black_hole_mass_exponent"] - np.log10(
            black_hole_mass_from_vel_disp(result["vel_disp"])
        )
        npt.assert_allclose(np.std(residual), 0.29, atol=0.05)
        npt.assert_allclose(np.mean(residual), 0.0, atol=0.05)

    def test_lognormal_eddington_distribution(self):
        """The lognormal option produces lower Eddington ratios than the
        power law for the same quasars and hosts."""
        np.random.seed(11)
        galaxy_cat = Table(
            {
                "z": np.random.uniform(0.1, 2.0, 20000),
                "vel_disp": 10 ** np.random.normal(2.15, 0.16, 20000),
            }
        )
        quasar_cat = Table(
            {
                "z": np.random.uniform(0.3, 1.8, 2000),
                "M_i": np.random.uniform(-26.0, -22.0, 2000),
            }
        )

        def median_log_lambda(distribution):
            result = QuasarHostMatch(
                quasar_catalog=quasar_cat,
                galaxy_catalog=galaxy_cat,
                eddington_ratio_distribution=distribution,
                min_candidates=50,
                progress=False,
            ).match()
            return np.median(np.log10(result["eddington_ratio"]))

        assert median_log_lambda("lognormal") < median_log_lambda("power_law")

    def test_unique_hosts(self):
        """With unique_hosts, no galaxy may host more than one quasar."""
        np.random.seed(5)
        galaxy_cat = Table(
            {
                "z": np.random.uniform(0.4, 0.6, 4000),
                "vel_disp": 10 ** np.random.normal(2.15, 0.16, 4000),
                "gal_id": np.arange(4000),
            }
        )
        quasar_cat = Table(
            {"z": np.full(500, 0.5), "M_i": np.random.uniform(-25.0, -22.0, 500)}
        )
        result = QuasarHostMatch(
            quasar_catalog=quasar_cat,
            galaxy_catalog=galaxy_cat,
            unique_hosts=True,
            progress=False,
        ).match()
        assert len(np.unique(result["gal_id"])) == len(result)

    def test_galaxy_type_selection(self, setup_catalogs):
        quasar_cat, galaxy_cat = setup_catalogs
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(),
            galaxy_catalog=galaxy_cat.copy(),
            galaxy_types=["red"],
            min_candidates=1,
            progress=False,
        )
        result = matcher.match()
        assert np.all(result["galaxy_type"] == "red")

        # a catalog round-tripped through FITS has a bytes galaxy_type column
        byte_types = galaxy_cat.copy()
        byte_types["galaxy_type"] = np.array([b"blue", b"red"])
        byte_result = QuasarHostMatch(
            quasar_catalog=quasar_cat.copy(),
            galaxy_catalog=byte_types,
            galaxy_types=["red"],
            min_candidates=1,
            progress=False,
        ).match()
        assert len(byte_result) == len(result)

        with pytest.raises(ValueError, match="No galaxy in the catalog"):
            QuasarHostMatch(
                quasar_catalog=quasar_cat.copy(),
                galaxy_catalog=galaxy_cat.copy(),
                galaxy_types=["green"],
                progress=False,
            ).match()

        with pytest.raises(ValueError, match="'galaxy_type' column"):
            galaxy_no_type = galaxy_cat.copy()
            galaxy_no_type.remove_column("galaxy_type")
            QuasarHostMatch(
                quasar_catalog=quasar_cat.copy(),
                galaxy_catalog=galaxy_no_type,
                galaxy_types=["red"],
                progress=False,
            ).match()

    def test_match_raises_error_if_no_vel_disp(self, setup_catalogs):
        quasar_cat, galaxy_cat = setup_catalogs
        galaxy_cat.remove_column("vel_disp")
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat, galaxy_catalog=galaxy_cat, progress=False
        )
        with pytest.raises(ValueError, match="must have 'vel_disp' column"):
            matcher.match()

    def test_no_host_candidates_in_range(self):
        """A quasar with no galaxies anywhere near its redshift is dropped."""
        quasar_cat = Table({"z": [4.5], "M_i": [-23.0]})
        galaxy_cat = Table(
            {"z": [0.5001, 0.4901], "vel_disp": [150.0, 200.0], "host_id": [1, 2]}
        )
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat, galaxy_catalog=galaxy_cat, progress=False
        )
        with pytest.warns(UserWarning, match="could not be assigned a host"):
            result = matcher.match()
        assert len(result) == 0
        assert matcher.n_rejected == 1

    def test_quasar_too_bright_for_any_host(self):
        """A quasar brighter than any available host could power is dropped."""
        quasar_cat = Table({"z": [0.5], "M_i": [-30.0]})
        galaxy_cat = Table({"z": [0.5, 0.5001], "vel_disp": [90.0, 95.0]})
        matcher = QuasarHostMatch(
            quasar_catalog=quasar_cat,
            galaxy_catalog=galaxy_cat,
            min_candidates=2,
            progress=False,
        )
        with pytest.warns(UserWarning, match="could not be assigned a host"):
            result = matcher.match()
        assert len(result) == 0
        assert matcher.n_rejected == 1

    def test_large_area(self):
        """Tests the matching against a skypy galaxy catalog."""
        quasar_cat = Table({"z": [0.5], "M_i": [-23.0], "ps_mag_i": [23.0]})
        sky_area = Quantity(1, unit="deg2")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        pipeline = SkyPyPipeline(
            skypy_config=None, sky_area=sky_area, filters=None, cosmo=cosmo
        )
        host_galaxy_catalog = vstack(
            [pipeline.red_galaxies, pipeline.blue_galaxies], join_type="exact"
        )
        f_vel_disp = vel_disp_abundance_matching(
            host_galaxy_catalog, z_max=0.5, sky_area=sky_area, cosmo=cosmo
        )
        host_galaxy_catalog["vel_disp"] = f_vel_disp(
            np.log10(host_galaxy_catalog["stellar_mass"])
        )

        result = QuasarHostMatch(
            quasar_catalog=quasar_cat,
            galaxy_catalog=host_galaxy_catalog,
            progress=False,
        ).match()

        assert isinstance(result, Table)
        assert len(result) == 1
