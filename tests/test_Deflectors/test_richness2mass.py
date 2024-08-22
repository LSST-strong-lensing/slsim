from slsim.Deflectors.richness2mass import mass_richness_relation


def test_mass_richness_relation():
    mass = mass_richness_relation(40, relation="simet2017")
    assert mass > 0

    mass = mass_richness_relation(40, relation="Abdullah2022")
    assert mass > 0
