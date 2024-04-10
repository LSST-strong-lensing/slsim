import pytest
import numpy.testing as npt
import os
from slsim.Deflectors.deflector import Deflector
from astropy.table import Table


class TestDeflector(object):
    """Testing the Deflector class."""

    def setup_method(self):
        path = os.path.dirname(__file__)
        module_path, _ = os.path.split(path)
        red_one = Table.read(
            os.path.join(module_path, "TestData/red_one_modified.fits"), format="fits"
        )
        self.deflector = Deflector(deflector_type="EPL", deflector_dict=red_one)

    def test_light_ellipticity(self):
        e1_light, e2_light = self.deflector.light_ellipticity
        assert pytest.approx(e1_light, rel=1e-3) == -0.05661955320450283
        assert pytest.approx(e2_light, rel=1e-3) == 0.08738390223219591

    def test_mass_ellipticity(self):
        e1_mass, e2_mass = self.deflector.mass_ellipticity
        assert pytest.approx(e1_mass, rel=1e-3) == -0.08434700688970058
        assert pytest.approx(e2_mass, rel=1e-3) == 0.09710653297997263

    def test_init(self):
        with npt.assert_raises(ValueError):
            Deflector(deflector_type="WRONG_MODEL", deflector_dict={})

    def test_magnitude(self):
        band = "g"
        deflector_magnitude = self.deflector.magnitude(band)
        assert isinstance(deflector_magnitude[0], float)
        assert pytest.approx(deflector_magnitude[0], rel=1e-3) == 26.4515655

    def test_redshift(self):
        z = self.deflector.redshift
        assert pytest.approx(z, rel=1e-3) == 0.9194649297646337

    def test_velocity_dispersion(self):
        sigma_v = self.deflector.velocity_dispersion(cosmo=None)
        assert pytest.approx(sigma_v, rel=1e-3) == 191.40371531030243

    def test_deflector_center(self):
        center = self.deflector.deflector_center
        assert isinstance(center[0], float)
        assert isinstance(center[1], float)

    def test_stellar_mass(self):
        stellar_mass = self.deflector.stellar_mass
        npt.assert_almost_equal(stellar_mass, 6.94160421e10, decimal=-3)
