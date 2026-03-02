import numpy as np
from numpy import testing as npt
from slsim.Util.catalog_util import normalize_features
import pytest

# catalog source matching is tested in test_Sources/testSourceTypes/test_catalog_source.py


def test_normalize_features():
    x = np.linspace(100, 200, 50)
    normalized_x = normalize_features(x, norm_type="minmax")
    npt.assert_allclose(normalized_x, np.linspace(0, 1, 50), atol=1e-15, rtol=1e-15)

    normalized_x = normalize_features(x, norm_type="zscore")
    npt.assert_allclose(
        normalized_x,
        np.linspace(-50, 50, 50) / 29.45075446869758,
        atol=1e-15,
        rtol=1e-15,
    )

    x = np.ones(10) * 51
    y = np.zeros_like(x)
    normalized_x = normalize_features(x, norm_type="minmax")
    npt.assert_equal(normalized_x, y)

    normalized_x = normalize_features(x, norm_type="zscore")
    npt.assert_equal(normalized_x, y)

    npt.assert_raises(ValueError, normalize_features, data=x, norm_type="incorrect")


if __name__ == "__main__":
    pytest.main()
