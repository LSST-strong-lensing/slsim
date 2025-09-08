import pytest
import numpy as np

from astropy import units as u
from astropy.table import Table
from numpy import testing as npt
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SourcePopulation.scotch_sources import (
    ScotchSources,
    galaxy_projected_eccentricity,
    _norm_band_names
)

def test_galaxy_projected_eccentricity():
    e1, e2 = galaxy_projected_eccentricity(0)
    assert e1 == 0
    assert e2 == 0

def test__norm_band_names():
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    norm_bands = _norm_band_names(bands)
    assert norm_bands == ['u', 'g', 'r', 'i', 'z', 'Y']

class TestScotchSources:
    