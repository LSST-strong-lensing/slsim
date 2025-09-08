import h5py
import pytest
import numpy as np

import slsim.Sources.SourcePopulation.scotch_sources as scotch_module


def test_norm_band_names():
    _norm = scotch_module._norm_band_names
    assert _norm(["U", "g", "Y", " y  "]) == ["u", "g", "Y", "Y"]

def test_galaxy_projected_eccentricity_deterministic():
    e1, e2 = scotch_module.galaxy_projected_eccentricity(ellipticity=0.0, rotation_angle=None)
    assert np.isclose(e1, 0.0) and np.isclose(e2, 0.0)

