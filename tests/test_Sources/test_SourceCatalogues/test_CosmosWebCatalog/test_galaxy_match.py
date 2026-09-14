import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from slsim.Sources.SourceCatalogues.CosmosWebCatalog.galaxy_match import (
    process_catalog,
)


@pytest.mark.parametrize(
    "size_columns, expected_angular_size",
    [
        pytest.param(
            {"sersic_radius": [2.0, 3.0], "axis_ratio": [0.25, 1.0]},
            [1.0, 3.0],
            id="circularize-sersic-radius",
        ),
        pytest.param(
            {"angular_size": [0.7, 1.5]},
            [0.7, 1.5],
            id="existing-angular-size-without-axis-ratio",
        ),
        pytest.param(
            {
                "sersic_radius": [2.0, 3.0],
                "axis_ratio": [0.25, 1.0],
                "angular_size": [9.0, 8.0],
            },
            [1.0, 3.0],
            id="sersic-radius-takes-precedence",
        ),
    ],
)
def test_process_catalog_sizes(tmp_path, size_columns, expected_angular_size):
    """Accept either size convention and convert the selected size to kpc."""
    catalog = Table({"id": [10, 20], "z": [0.5, 1.0], **size_columns})
    catalog.write(tmp_path / "COSMOSWeb_galaxy_catalog.fits", format="fits")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    result = process_catalog(cosmo, str(tmp_path))

    np.testing.assert_allclose(result["angular_size"], expected_angular_size)
    assert result["angular_size"].unit == u.arcsec
    expected_physical_size = (
        np.asarray(expected_angular_size)
        * u.arcsec
        * cosmo.kpc_proper_per_arcmin(catalog["z"])
    ).to_value(u.kpc)
    np.testing.assert_allclose(result["physical_size"], expected_physical_size)
    assert result["physical_size"].unit == u.kpc
    np.testing.assert_array_equal(result["id"], catalog["id"])
    np.testing.assert_array_equal(result["z"], catalog["z"])


def test_process_catalog_missing_size(tmp_path):
    """Reject catalogs with neither supported size column."""
    catalog = Table({"z": [0.5], "axis_ratio": [0.25]})
    catalog.write(tmp_path / "COSMOSWeb_galaxy_catalog.fits", format="fits")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with pytest.raises(
        ValueError,
        match="must contain either a column named 'sersic_radius' or 'angular_size'",
    ):
        process_catalog(cosmo, str(tmp_path))
