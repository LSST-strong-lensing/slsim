import numpy as np
from astropy.table import Table
from slsim.Sources.Events.BNSMerger.bns_mass_distribution import BNSMassDistribution


def test_assign_masses():
    population = Table(
        {
            "event_id": [0, 1, 2],
            "z": [0.1, 0.2, 0.3],
        }
    )

    mass_distribution = BNSMassDistribution()
    result = mass_distribution.assign_masses(population)

    assert len(result) == len(population)

    for name in [
        "mass_recycled",
        "mass_slow",
        "mass_1",
        "mass_2",
        "Mchirp",
        "q",
    ]:
        assert name in result.colnames

    # Check the sampled mass ranges.
    mr = result["mass_recycled"]
    ms = result["mass_slow"]

    assert np.all((mr >= 0.8) & (mr <= 2.0))
    assert np.all((ms >= 1.16) & (ms <= 1.42))

    # Check the component-mass ordering.
    m1 = result["mass_1"]
    m2 = result["mass_2"]

    assert np.all(m1 >= m2)
    np.testing.assert_allclose(m1, np.maximum(mr, ms))
    np.testing.assert_allclose(m2, np.minimum(mr, ms))

    # Check the chirp mass and mass ratio.
    expected_mchirp = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    expected_q = m2 / m1

    np.testing.assert_allclose(result["Mchirp"], expected_mchirp)
    np.testing.assert_allclose(result["q"], expected_q)
