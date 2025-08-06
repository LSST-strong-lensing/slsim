from slsim.Lenses.selection import object_cut
import numpy as np
from astropy.table import Table
import pytest

sample1 = Table({"mag_i": np.linspace(20, 30, 20), "z": np.linspace(0.01, 2, 20)})
sample2 = Table({"mag_i": np.array([23]), "z": np.array([0.5])})
sample3 = Table({"mag_i": np.array([30]), "z": np.array([1.2])})
sample = [sample2, sample3]


def test_object_cut():
    sample_cut = object_cut(
        sample1,
        z_min=0,
        z_max=1,
        band=None,
        band_max=40,
        list_type="astropy_table",
        object_type="extended",
    )
    sample_cut2 = object_cut(
        sample,
        z_min=0,
        z_max=1,
        band="i",
        band_max=26,
        list_type="list",
        object_type="extended",
    )
    assert max(sample_cut["z"]) < 1
    assert sample_cut2[0]["mag_i"] == 23
    with pytest.raises(ValueError):
        object_cut(
            sample1,
            z_min=0,
            z_max=1,
            band=None,
            band_max=40,
            list_type="astropy_table",
            object_type="source",
        )


if __name__ == "__main__":
    pytest.main()
