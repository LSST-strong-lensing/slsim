from slsim.Sources.QuasarCatalog import quasar_plus_galaxies
from astropy.units import Quantity
import pytest


def test_quasar_plus_galaxies():
    catalog = quasar_plus_galaxies.quasar_galaxies_simple(
        m_min=17,
        m_max=23,
        amp_min=0.9,
        amp_max=1.3,
        freq_min=0.5,
        freq_max=1.5,
        sky_area=Quantity(value=0.05, unit="deg2"),
    )
    column_names = [
        "z",
        "M",
        "coeff",
        "ellipticity",
        "physical_size",
        "stellar_mass",
        "angular_size",
        "mag_g",
        "mag_r",
        "mag_i",
        "mag_z",
        "mag_Y",
        "ps_mag_r",
        "ps_mag_g",
        "ps_mag_i",
        "amp",
        "freq",
    ]
    assert catalog.colnames[0] == column_names[0]
    assert catalog.colnames[1] == column_names[1]
    assert catalog.colnames[2] == column_names[2]
    assert catalog.colnames[3] == column_names[3]
    assert catalog.colnames[4] == column_names[4]
    assert catalog.colnames[5] == column_names[5]
    assert catalog.colnames[6] == column_names[6]
    assert catalog.colnames[7] == column_names[7]
    assert catalog.colnames[8] == column_names[8]
    assert catalog.colnames[9] == column_names[9]
    assert catalog.colnames[10] == column_names[10]
    assert catalog.colnames[11] == column_names[11]
    assert catalog.colnames[12] == column_names[12]
    assert catalog.colnames[13] == column_names[13]
    assert catalog.colnames[14] == column_names[14]
    assert catalog.colnames[15] == column_names[15]
    assert catalog.colnames[16] == column_names[16]


if __name__ == "__main__":
    pytest.main()
