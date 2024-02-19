import pytest
import numpy as np
import astropy.units as u
from slsim.Sources.simple_supernova_lightcurve import SimpleSupernovaLightCurve

@pytest.fixture
def simple_light_curve():
    cosmo = None
    return SimpleSupernovaLightCurve(cosmo)

def test_generate_light_curve(simple_light_curve):
    redshift = 0.5
    abs_mag = -19.5
    num_points = 50
    lightcurve_time = 50 * u.day
    band = "r"

    time, mags = simple_light_curve.generate_light_curve(redshift, abs_mag, num_points, lightcurve_time, band)

    assert len(time) == num_points
    assert len(mags) == num_points
    assert np.all(np.isfinite(mags))
def test_generate_light_curve_invalid_band(simple_light_curve):
    # Test for invalid band
    with pytest.raises(ValueError):
        simple_light_curve.generate_light_curve(0.5, -19.5, band="invalid_band")
