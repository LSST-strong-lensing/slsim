import numpy as np
import numpy.testing as npt
import pytest
from astropy import units as u

from slsim.Sources.SourcePopulation.catalog_preparation import prepare_source_catalog
from slsim.Util.param_util import galaxy_size_redshift_evolution

raw_catalog = {"Z": [1.0, 2.0, 3.0], "FLUX_I": [3631.0, 0.0, -1.0]}


def test_prepare_source_catalog():
    catalog = prepare_source_catalog(
        raw_catalog,
        column_mapping={"Z": "z"},
        flux_columns={"FLUX_I": "i"},
        flux_unit=u.Jy,
        fill_missing=["physical_size", "ellipticity"],
    )
    npt.assert_array_equal(catalog["z"], raw_catalog["Z"])
    npt.assert_array_equal(catalog["mag_i"], [0, np.inf, np.inf])
    npt.assert_allclose(
        catalog["physical_size"], galaxy_size_redshift_evolution(catalog["z"])
    )
    assert np.all((catalog["ellipticity"] >= 0) & (catalog["ellipticity"] <= 1))


def test_prepare_source_catalog_errors():
    with pytest.raises(u.UnitConversionError):
        prepare_source_catalog(raw_catalog, flux_columns={"FLUX_I": "i"})
    with pytest.raises(ValueError, match="No model to fill"):
        prepare_source_catalog(raw_catalog, fill_missing=["angular_size"])
