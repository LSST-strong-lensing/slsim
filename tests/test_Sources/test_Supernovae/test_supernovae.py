from slsim.Sources.Supernovae.supernova import Supernova
import numpy.testing as npt
import pytest


@pytest.fixture
def Supernova_class():
    SN = Supernova(
        source="salt3-nir",
        redshift=1.2,
        sn_type="Ia",
        absolute_mag=-19.3,
        absolute_mag_band="bessellb",
        mag_zpsys="AB",
    )

    return SN


def test_supernova_mag(Supernova_class):
    mag = Supernova_class.get_apparent_magnitude(time=0, band="lsstr")
    assert mag > 0

    npt.assert_warns(
        UserWarning, Supernova_class.get_apparent_magnitude, time=0, band="lsstg"
    )


if __name__ == "__main__":
    pytest.main()
