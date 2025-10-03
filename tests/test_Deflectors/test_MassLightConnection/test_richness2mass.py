from slsim.Deflectors.MassLightConnection.richness2mass import mass_richness_relation
import pytest


def test_mass_richness_relation():
    for _ in range(100000):
        # random mass around richness 40
        mass = mass_richness_relation(40, relation="simet2017")
        assert (mass >= 1e12) and (mass < 1e15)

        mass = mass_richness_relation(40, relation="Abdullah2022")
        assert (mass >= 1e12) and (mass < 1e15)


def test_unknown_mass_richness_relation():
    with pytest.raises(ValueError):
        mass_richness_relation(40, relation="unknown")


if __name__ == "__main__":
    pytest.main()
