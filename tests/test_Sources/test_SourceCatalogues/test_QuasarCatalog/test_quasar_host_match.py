import numpy as np
import numpy.testing as npt
import pytest
from astropy.table import Table

from slsim.Sources.SourceCatalogues.QuasarCatalog.quasar_host_match import (
    BLACK_HOLE_MASS_RELATIONS,
    ERDF_LOCATION,
    ERDF_WIDTH,
    L3000_ZERO_POINT,
    L_EDDINGTON_PER_MSUN,
    BOLOMETRIC_CORRECTION_3000,
    QuasarHostMatch,
    log_black_hole_mass,
    log_bolometric_luminosity,
)


def galaxy_catalog(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    return Table(
        {
            "z": rng.uniform(0.1, 2.0, n),
            "vel_disp": 10 ** rng.normal(2.15, 0.25, n),
            "stellar_mass": 10 ** rng.normal(10.5, 0.6, n),
            "galaxy_type": rng.choice(["red", "blue"], n),
        }
    )


def quasar_catalog(n=2000, seed=4, m_i=(-26.0, -23.0)):
    rng = np.random.default_rng(seed)
    return Table({"z": rng.uniform(0.3, 1.8, n), "M_i": rng.uniform(*m_i, n)})


def matcher(quasars=None, galaxies=None, seed=1, **kwargs):
    return QuasarHostMatch(
        quasar_catalog=quasar_catalog() if quasars is None else quasars,
        galaxy_catalog=galaxy_catalog() if galaxies is None else galaxies,
        rng=np.random.default_rng(seed),
        progress=False,
        **kwargs,
    )


class TestLogBlackHoleMass:
    def test_red_follows_kormendy_ho(self):
        log_mass, scatter = log_black_hole_mass(["red", "red"], vel_disp=[200.0, 150.0])
        npt.assert_allclose(log_mass[0], np.log10(0.309e9))
        npt.assert_allclose(log_mass[1] - log_mass[0], 4.38 * np.log10(150 / 200))
        npt.assert_allclose(scatter, 0.29)

    def test_blue_follows_reines_volonteri(self):
        log_mass, scatter = log_black_hole_mass(
            ["blue", "blue"], stellar_mass=[1e11, 1e10]
        )
        npt.assert_allclose(log_mass, [7.45, 7.45 - 1.05])
        npt.assert_allclose(scatter, 0.24)

    def test_mixed_types_use_their_own_relation(self):
        log_mass, scatter = log_black_hole_mass(
            ["red", "blue"], vel_disp=[200.0, np.nan], stellar_mass=[np.nan, 1e11]
        )
        npt.assert_allclose(log_mass, [np.log10(0.309e9), 7.45])
        npt.assert_allclose(scatter, [0.29, 0.24])

    def test_bytes_galaxy_type(self):
        """FITS round-trips strings as bytes."""
        log_mass, _ = log_black_hole_mass(np.array([b"red"]), vel_disp=[200.0])
        npt.assert_allclose(log_mass, np.log10(0.309e9))

    def test_missing_property(self):
        with pytest.raises(ValueError, match="vel_disp"):
            log_black_hole_mass(["red"], stellar_mass=[1e11])

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="green"):
            log_black_hole_mass(["green"], vel_disp=[200.0])


class TestLogBolometricLuminosity:
    def test_zero_point_and_bolometric_correction(self):
        m_i = np.array([-28.0, -26.0, -24.0])
        expected = L3000_ZERO_POINT - 0.4 * m_i + np.log10(BOLOMETRIC_CORRECTION_3000)
        npt.assert_allclose(log_bolometric_luminosity(m_i), expected)
        # an M_i = -26 quasar sits near log L_bol = 46.4 in Wu & Shen (2022)
        npt.assert_allclose(expected[1], 46.4, atol=0.1)

    def test_scatter_is_unbiased_in_the_log(self):
        scattered = log_bolometric_luminosity(
            np.full(100000, -26.0), scatter=0.1, rng=np.random.default_rng(0)
        )
        npt.assert_allclose(
            scattered.mean(), log_bolometric_luminosity(-26.0), atol=0.01
        )
        npt.assert_allclose(scattered.std(), 0.1, atol=0.01)


class TestQuasarHostMatch:
    def test_match_is_self_consistent(self):
        """The black hole mass and Eddington ratio reproduce the bolometric
        luminosity."""
        result = matcher().match()
        assert len(result) > 0
        npt.assert_allclose(
            result["log_bolometric_luminosity"],
            result["black_hole_mass_exponent"]
            + np.log10(L_EDDINGTON_PER_MSUN)
            + np.log10(result["eddington_ratio"]),
            atol=1e-10,
        )

    def test_match_samples_the_analytic_posterior(self):
        """Host frequencies and conditional Eddington ratios follow the closed-
        form posterior for two red hosts of known mean mass."""
        red = BLACK_HOLE_MASS_RELATIONS["red"]
        mean_log_mass = np.array([8.8, 9.2])
        n = 4000
        result = matcher(
            Table({"z": np.full(n, 0.5), "M_i": np.full(n, -25.0)}),
            Table(
                {
                    "z": [0.5, 0.5],
                    "vel_disp": 200
                    * 10 ** ((mean_log_mass - red["intercept"]) / red["slope"]),
                    "galaxy_type": ["red", "red"],
                    "host_id": [0, 1],
                }
            ),
            seed=3,
            min_candidates=2,
            bolometric_correction_scatter=0,
        ).match()

        c = log_bolometric_luminosity(-25.0) - np.log10(L_EDDINGTON_PER_MSUN)
        variance = red["scatter"] ** 2 + ERDF_WIDTH**2
        weight = np.exp(-0.5 * (c - mean_log_mass - ERDF_LOCATION) ** 2 / variance)
        npt.assert_allclose(
            np.bincount(result["host_id"], minlength=2) / len(result),
            weight / weight.sum(),
            atol=0.025,
        )

        log_lambda = np.log10(result["eddington_ratio"])
        v = 1 / (1 / ERDF_WIDTH**2 + 1 / red["scatter"] ** 2)
        for host_id in (0, 1):
            selected = result["host_id"] == host_id
            expected_mean = v * (
                ERDF_LOCATION / ERDF_WIDTH**2
                + (c - mean_log_mass[host_id]) / red["scatter"] ** 2
            )
            npt.assert_allclose(log_lambda[selected].mean(), expected_mean, atol=0.02)
            npt.assert_allclose(log_lambda[selected].std(), np.sqrt(v), atol=0.02)

    def test_rng_makes_the_catalog_reproducible(self):
        def run():
            return matcher(
                quasar_catalog(n=200), galaxy_catalog(n=5000), seed=5
            ).match()

        npt.assert_array_equal(
            run()["black_hole_mass_exponent"], run()["black_hole_mass_exponent"]
        )

    def test_rejects_quasars_no_host_can_produce(self):
        """A quasar far too luminous for any available black hole is dropped,
        with a warning saying where the dropped ones sit."""
        m = matcher(
            Table({"z": [0.5], "M_i": [-32.0]}),
            Table({"z": [0.5], "vel_disp": [100.0], "galaxy_type": ["red"]}),
            min_candidates=1,
        )
        with pytest.warns(UserWarning, match="were dropped"):
            result = m.match()
        assert len(result) == 0
        assert m.rejected_indices == [0]

    def test_match_keeps_the_host_columns(self):
        result = matcher(
            Table({"z": [0.5], "M_i": [-23.0]}),
            Table(
                {
                    "z": [0.5001, 0.4999],
                    "vel_disp": [150.0, 200.0],
                    "stellar_mass": [1e11, 2e11],
                    "galaxy_type": ["blue", "red"],
                    "host_id": [1, 2],
                }
            ),
            min_candidates=2,
        ).match()
        for column in [
            "z",
            "M_i",
            "host_id",
            "vel_disp",
            "stellar_mass",
            "black_hole_mass_exponent",
            "eddington_ratio",
            "log_bolometric_luminosity",
        ]:
            assert column in result.colnames
        assert result["host_id"][0] in [1, 2]

    def test_bytes_galaxy_type_end_to_end(self):
        galaxies = galaxy_catalog(n=5000)
        galaxies["galaxy_type"] = np.char.encode(
            np.asarray(galaxies["galaxy_type"], dtype=str)
        )
        assert len(matcher(quasar_catalog(n=100), galaxies).match()) > 0

    def test_galaxies_without_a_usable_property_are_dropped(self):
        result = matcher(
            Table({"z": [0.5], "M_i": [-23.0]}),
            Table(
                {
                    "z": [0.5, 0.5],
                    "stellar_mass": [0.0, 1e11],
                    "galaxy_type": ["blue", "blue"],
                    "host_id": [1, 2],
                }
            ),
            min_candidates=1,
        ).match()
        assert list(result["host_id"]) == [2]

    def test_no_usable_galaxy(self):
        with pytest.raises(ValueError, match="usable black hole mass"):
            matcher(
                Table({"z": [0.5], "M_i": [-23.0]}),
                Table({"z": [0.5], "stellar_mass": [0.0], "galaxy_type": ["blue"]}),
            ).match()

    def test_missing_quasar_column(self):
        with pytest.raises(ValueError, match="M_i"):
            matcher(Table({"z": [0.5]}), galaxy_catalog(n=10)).match()

    def test_missing_galaxy_type_column(self):
        with pytest.raises(ValueError, match="galaxy_type"):
            matcher(
                quasar_catalog(n=10), Table({"z": [0.5], "vel_disp": [200.0]})
            ).match()
