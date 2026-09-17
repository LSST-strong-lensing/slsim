"""Check opt-in subtraction, native-band ordering, and cached reuse."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SourceTypes.catalog_source import CatalogSource


@pytest.mark.parametrize("catalog_type, count", [("HST_COSMOS", 1), ("COSMOS_WEB", 4)])
@pytest.mark.parametrize("enabled", [False, True])
def test_background_native_bands(monkeypatch, catalog_type, count, enabled):
    attr = (
        "processed_hst_cosmos_catalog" if count == 1 else "processed_cosmos_web_catalog"
    )
    monkeypatch.setattr(CatalogSource, attr, None, raising=False)
    source = CatalogSource(
        angular_size=0.3,
        e1=0.1,
        e2=0.0,
        n_sersic=1,
        cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        catalog_type=catalog_type,
        catalog_path="unused",
        subtract_background=enabled,
        z=1,
        mag_i=22,
        mag_r=22,
        center_x=0,
        center_y=0,
    )
    y, x = np.indices((80, 80))
    galaxy = 20 * np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / 40)
    images = [galaxy + 2 * (i + 1) for i in range(count)]
    originals = [a.copy() for a in images]
    monkeypatch.setattr(
        source, "_match_source", lambda **kw: (images, 0.03, 0, {"NOISE_MEAN": 2.0})
    )
    _, first = source.kwargs_extended_light("i")
    saved = [a.copy() for a in source._image_list]
    source.kwargs_extended_light("r")
    for i, (raw, current) in enumerate(zip(originals, source._image_list)):
        np.testing.assert_array_equal(images[i], raw)
        np.testing.assert_array_equal(current, saved[i])
        np.testing.assert_allclose(current, galaxy if enabled else raw, atol=1e-6)
    expected = source._image_list[0 if count == 1 else 2]
    np.testing.assert_array_equal(first[0]["image"], expected)
    assert (source.background_diagnostics is not None) == enabled
