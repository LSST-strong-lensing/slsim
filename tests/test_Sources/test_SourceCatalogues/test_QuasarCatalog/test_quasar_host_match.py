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
    RUNNOE12_ZETA_3000,
    QuasarHostMatch,
    _conditional_eddington_parameters,
    _draw_from_log_weights,
    _host_log_weights,
    black_hole_mass,
    bolometric_luminosity,
    log_bolometric_luminosity,
)


def galaxy_catalog(n=20000, seed=3):
    """A mixed red and blue host catalog spanning a broad range of mass."""
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
    return Table({"z": rng.uniform(0.3, 1.8, n), "M_i": rng.uniform(m_i[0], m_i[1], n)})


class TestBlackHoleMass:
    def test_red_follows_kormendy_ho(self):
        """Kormendy & Ho (2013) eq.

        7: M_BH = 0.309e9 Msun at sigma = 200 km/s.
        """
        mass, scatter = black_hole_mass(["red"], vel_disp=[200.0])
        npt.assert_allclose(mass, 0.309e9)
        npt.assert_allclose(scatter, 0.29)

        mass, _ = black_hole_mass(["red", "red"], vel_disp=[200.0, 150.0])
        npt.assert_allclose(mass[1] / mass[0], (150 / 200) ** 4.38)

    def test_blue_follows_reines_volonteri(self):
        """Reines & Volonteri (2015) eq.

        5: log M_BH = 7.45 at M* = 1e11.
        """
        mass, scatter = black_hole_mass(["blue"], stellar_mass=[1e11])
        npt.assert_allclose(np.log10(mass), 7.45)
        npt.assert_allclose(scatter, 0.24)

        mass, _ = black_hole_mass(["blue", "blue"], stellar_mass=[1e11, 1e10])
        npt.assert_allclose(np.log10(mass[0] / mass[1]), 1.05)

    def test_blue_hosts_smaller_black_holes_than_red(self):
        """The disc relation sits below the bulge one, which is the whole point
        of applying each only where it is calibrated."""
        red, _ = black_hole_mass(["red"], vel_disp=[150.0])
        blue, _ = black_hole_mass(["blue"], stellar_mass=[1e11])
        assert blue < red

    def test_mixed_types_use_their_own_relation(self):
        mass, scatter = black_hole_mass(
            ["red", "blue"], vel_disp=[200.0, np.nan], stellar_mass=[np.nan, 1e11]
        )
        npt.assert_allclose(np.log10(mass), [np.log10(0.309e9), 7.45])
        npt.assert_allclose(scatter, [0.29, 0.24])

    def test_bytes_galaxy_type(self):
        """FITS round-trips strings as bytes, which must still be matched."""
        mass, _ = black_hole_mass(np.array([b"red"]), vel_disp=[200.0])
        npt.assert_allclose(mass, 0.309e9)

    def test_missing_property(self):
        with pytest.raises(ValueError, match="vel_disp"):
            black_hole_mass(["red"], stellar_mass=[1e11])

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="green"):
            black_hole_mass(["green"], vel_disp=[200.0])


class TestAnalyticPosterior:
    def test_host_weights_are_the_gaussian_convolution(self):
        c = 8.0
        means = np.array([7.5, 8.5])
        scatters = np.array([0.2, 0.4])
        log_weight = _host_log_weights(c, means, scatters, -1.0, 0.3)
        variance = scatters**2 + 0.3**2
        expected = -0.5 * (np.log(variance) + (c - means + 1.0) ** 2 / variance)
        npt.assert_allclose(log_weight, expected)

    def test_conditional_eddington_distribution(self):
        mean, variance = _conditional_eddington_parameters(
            c=8.2,
            mean_log_mass=9.0,
            mass_scatter=0.4,
            location=-1.0,
            width=0.3,
        )
        expected_variance = 1 / (1 / 0.3**2 + 1 / 0.4**2)
        expected_mean = expected_variance * (-1.0 / 0.3**2 + (8.2 - 9.0) / 0.4**2)
        npt.assert_allclose(variance, expected_variance)
        npt.assert_allclose(mean, expected_mean)

    def test_draw_stabilizes_extreme_log_weights(self):
        rng = np.random.default_rng(0)
        draws = np.array(
            [_draw_from_log_weights([-1000, -1001, -np.inf], rng) for _ in range(10000)]
        )
        assert not np.any(draws == 2)
        npt.assert_allclose(np.mean(draws == 0), 1 / (1 + np.exp(-1)), atol=0.02)


class TestBolometricLuminosity:
    def test_scales_with_magnitude(self):
        m_i = np.array([-28.0, -26.0, -24.0])
        npt.assert_allclose(np.diff(np.log10(bolometric_luminosity(m_i))), -0.8)

    def test_applies_the_dr16q_zero_point_and_the_runnoe_correction(self):
        m_i = -26.0
        log_l3000 = L3000_ZERO_POINT - 0.4 * m_i
        npt.assert_allclose(
            np.log10(bolometric_luminosity(m_i)),
            log_l3000 + np.log10(RUNNOE12_ZETA_3000),
        )
        # an M_i = -26 quasar sits near log L_bol = 46.4 in Wu & Shen (2022)
        npt.assert_allclose(np.log10(bolometric_luminosity(m_i)), 46.4, atol=0.1)

    def test_scatter_is_unbiased_in_the_log(self):
        rng = np.random.default_rng(0)
        m_i = np.full(100000, -26.0)
        scattered = np.log10(bolometric_luminosity(m_i, scatter=0.1, rng=rng))
        npt.assert_allclose(
            scattered.mean(), np.log10(bolometric_luminosity(-26.0)), atol=0.01
        )
        npt.assert_allclose(scattered.std(), 0.1, atol=0.01)

    def test_log_variant_avoids_a_round_trip(self):
        npt.assert_allclose(
            log_bolometric_luminosity([-26.0, -24.0]),
            np.log10(bolometric_luminosity([-26.0, -24.0])),
        )


class TestQuasarHostMatch:
    def test_match_is_self_consistent(self):
        """The assigned black hole mass and Eddington ratio must reproduce the
        bolometric luminosity implied by the quasar magnitude."""
        result = QuasarHostMatch(
            quasar_catalog=quasar_catalog(),
            galaxy_catalog=galaxy_catalog(),
            rng=np.random.default_rng(1),
        ).match(progress=False)
        assert len(result) > 0

        log_l_edd = (
            result["black_hole_mass_exponent"]
            + np.log10(L_EDDINGTON_PER_MSUN)
            + np.log10(result["eddington_ratio"])
        )
        npt.assert_allclose(result["log_bolometric_luminosity"], log_l_edd, atol=1e-10)

    def test_match_samples_the_analytic_posterior(self):
        """Host frequencies and conditional Eddington ratios follow the closed-
        form posterior."""
        mean_log_mass = np.array([8.8, 9.2])
        intercept = BLACK_HOLE_MASS_RELATIONS["red"]["intercept"]
        slope = BLACK_HOLE_MASS_RELATIONS["red"]["slope"]
        velocity_dispersion = 200 * 10 ** ((mean_log_mass - intercept) / slope)
        n_quasars = 4000
        quasars = Table(
            {"z": np.full(n_quasars, 0.5), "M_i": np.full(n_quasars, -25.0)}
        )
        galaxies = Table(
            {
                "z": [0.5, 0.5],
                "vel_disp": velocity_dispersion,
                "galaxy_type": ["red", "red"],
                "host_id": [0, 1],
            }
        )
        result = QuasarHostMatch(
            quasar_catalog=quasars,
            galaxy_catalog=galaxies,
            min_candidates=2,
            bolometric_correction_scatter=0,
            rng=np.random.default_rng(3),
        ).match(progress=False)

        c = log_bolometric_luminosity(-25.0) - np.log10(L_EDDINGTON_PER_MSUN)
        scatter = BLACK_HOLE_MASS_RELATIONS["red"]["scatter"]
        log_weight = _host_log_weights(
            c, mean_log_mass, np.full(2, scatter), ERDF_LOCATION, ERDF_WIDTH
        )
        expected_host_probability = np.exp(log_weight - np.max(log_weight))
        expected_host_probability /= expected_host_probability.sum()
        observed_host_probability = np.bincount(result["host_id"], minlength=2) / len(
            result
        )
        npt.assert_allclose(
            observed_host_probability, expected_host_probability, atol=0.025
        )

        log_eddington_ratio = np.log10(result["eddington_ratio"])
        for host_id in (0, 1):
            expected_mean, expected_variance = _conditional_eddington_parameters(
                c, mean_log_mass[host_id], scatter, ERDF_LOCATION, ERDF_WIDTH
            )
            selected = result["host_id"] == host_id
            npt.assert_allclose(
                log_eddington_ratio[selected].mean(), expected_mean, atol=0.02
            )
            npt.assert_allclose(
                log_eddington_ratio[selected].std(),
                np.sqrt(expected_variance),
                atol=0.02,
            )

    def test_rng_makes_the_catalog_reproducible(self):
        def run():
            return QuasarHostMatch(
                quasar_catalog=quasar_catalog(n=200),
                galaxy_catalog=galaxy_catalog(n=5000),
                rng=np.random.default_rng(5),
            ).match(progress=False)

        npt.assert_array_equal(
            run()["black_hole_mass_exponent"], run()["black_hole_mass_exponent"]
        )

    def test_rejects_quasars_no_host_can_produce(self):
        """A quasar far too luminous for any available black hole is dropped,
        with a warning saying where the dropped ones sit."""
        matcher = QuasarHostMatch(
            quasar_catalog=Table({"z": [0.5], "M_i": [-32.0]}),
            galaxy_catalog=Table(
                {
                    "z": [0.5],
                    "vel_disp": [100.0],
                    "stellar_mass": [1e10],
                    "galaxy_type": ["red"],
                }
            ),
            min_candidates=1,
        )
        with pytest.warns(UserWarning, match="were dropped"):
            result = matcher.match(progress=False)
        assert len(result) == 0
        assert matcher.n_rejected == 1
        assert matcher.rejected_indices == [0]

    def test_match_keeps_the_host_columns(self):
        result = QuasarHostMatch(
            quasar_catalog=Table({"z": [0.5], "M_i": [-23.0]}),
            galaxy_catalog=Table(
                {
                    "z": [0.5001, 0.4999],
                    "vel_disp": [150.0, 200.0],
                    "stellar_mass": [1e11, 2e11],
                    "galaxy_type": ["blue", "red"],
                    "host_id": [1, 2],
                }
            ),
            min_candidates=2,
            rng=np.random.default_rng(6),
        ).match(progress=False)

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
        """A host catalog round-tripped through FITS must still match."""
        galaxies = galaxy_catalog(n=5000)
        galaxies["galaxy_type"] = np.char.encode(
            np.asarray(galaxies["galaxy_type"], dtype=str)
        )
        result = QuasarHostMatch(
            quasar_catalog=quasar_catalog(n=100),
            galaxy_catalog=galaxies,
            rng=np.random.default_rng(7),
        ).match(progress=False)
        assert len(result) > 0

    def test_galaxies_without_a_usable_property_are_dropped(self):
        """A blue galaxy with no stellar mass carries no black hole
        information, so it cannot be a host and the other galaxy must be
        chosen."""
        result = QuasarHostMatch(
            quasar_catalog=Table({"z": [0.5], "M_i": [-23.0]}),
            galaxy_catalog=Table(
                {
                    "z": [0.5, 0.5],
                    "vel_disp": [150.0, 150.0],
                    "stellar_mass": [0.0, 1e11],
                    "galaxy_type": ["blue", "blue"],
                    "host_id": [1, 2],
                }
            ),
            min_candidates=1,
            rng=np.random.default_rng(8),
        ).match(progress=False)
        assert list(result["host_id"]) == [2]

    def test_no_usable_galaxy(self):
        with pytest.raises(ValueError, match="usable black hole mass"):
            QuasarHostMatch(
                quasar_catalog=Table({"z": [0.5], "M_i": [-23.0]}),
                galaxy_catalog=Table(
                    {
                        "z": [0.5],
                        "stellar_mass": [0.0],
                        "galaxy_type": ["blue"],
                    }
                ),
            ).match()

    def test_missing_quasar_column(self):
        with pytest.raises(ValueError, match="M_i"):
            QuasarHostMatch(
                quasar_catalog=Table({"z": [0.5]}),
                galaxy_catalog=galaxy_catalog(n=10),
            ).match()

    def test_missing_galaxy_type_column(self):
        with pytest.raises(ValueError, match="galaxy_type"):
            QuasarHostMatch(
                quasar_catalog=quasar_catalog(n=10),
                galaxy_catalog=Table({"z": [0.5], "vel_disp": [200.0]}),
            ).match()
