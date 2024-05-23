import pytest
import numpy as np
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SupernovaeCatalog.supernovae_sample import SupernovaeCatalog

sn_type = "Ia"
band_list = ["r"]
lightcurve_time = np.linspace(-20, 100, 500)
absolute_mag_band = "bessellb"
mag_zpsys = "AB"
skypy_config = None
sky_area = Quantity(0.0001, unit="deg2")
absolute_mag = None
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


class TestSupernovaeCatalog:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.supernovae_catalog = SupernovaeCatalog(
            sn_type=sn_type,
            band_list=band_list,
            lightcurve_time=lightcurve_time,
            absolute_mag_band=absolute_mag_band,
            mag_zpsys=mag_zpsys,
            cosmo=cosmo,
            skypy_config=skypy_config,
            sky_area=sky_area,
            absolute_mag=absolute_mag,
        )

    def test_host_galaxy_catalog(self):
        result = self.supernovae_catalog.host_galaxy_catalog()
        assert all(result["z"] <= 0.9329)

    def test_supernovae_catalog(self):
        result = self.supernovae_catalog.supernovae_catalog()
        result2 = self.supernovae_catalog.supernovae_catalog(host_galaxy=False)

        assert "MJD" in result.colnames
        assert "ps_mag_r" in result.colnames
        assert "z" in result.colnames
        assert len(result2.colnames) == 3

    def test_supernovae_host_galaxy_offset(self):
        ra_off, dec_off = self.supernovae_catalog.supernovae_host_galaxy_offset(5)
        assert max(ra_off) <= 5
        assert min(ra_off) >= -5
        assert max(dec_off) <= 5
        assert min(dec_off) >= -5


if __name__ == "__main__":
    pytest.main()
