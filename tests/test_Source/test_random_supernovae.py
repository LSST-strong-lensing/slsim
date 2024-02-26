from slsim.Sources.random_supernovae import RandomizedSupernova
import pytest


@pytest.fixture
def Random_Supernova_Ia_class():
    SN = RandomizedSupernova(sn_type="Ia", redshift=1.0)
    return SN


@pytest.fixture
def Random_Supernova_CC_class():
    SN = RandomizedSupernova(sn_type="Ib", redshift=1.0)
    return SN


def test_random_supernova_Ia_mag(Random_Supernova_Ia_class):
    mag = Random_Supernova_Ia_class.get_apparent_magnitude(time=0, band="lsstr")
    assert mag > 0


def test_random_supernova_CC_mag(Random_Supernova_CC_class):
    mag = Random_Supernova_CC_class.get_apparent_magnitude(time=0, band="lsstr")
    assert mag > 0


if __name__ == "__main__":
    pytest.main()
