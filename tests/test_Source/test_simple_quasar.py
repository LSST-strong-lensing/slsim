from slsim.Sources.QuasarCatalog import simple_quasar
import pytest


def test_simple_quasar():
    kwargs_quasars = {
        "num_quasars": 50000,
        "z_min": 0.1,
        "z_max": 5,
        "m_min": 17,
        "m_max": 25,
        "amp_min": 0.9,
        "amp_max": 1.3,
        "freq_min": 0.5,
        "freq_max": 1.5,
    }
    catalog = simple_quasar.quasar_catalog_simple(**kwargs_quasars)
    column_names = ["z", "ps_mag_r", "ps_mag_g", "ps_mag_i"]
    assert catalog.colnames[0] == column_names[0]
    assert catalog.colnames[1] == column_names[1]
    assert catalog.colnames[2] == column_names[2]
    assert catalog.colnames[3] == column_names[3]


if __name__ == "__main__":
    pytest.main()
