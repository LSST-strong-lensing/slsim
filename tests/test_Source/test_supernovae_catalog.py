import pytest
import numpy as np
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SupernovaeCatalog.supernovae_sample import SupernovaeCatalog
from slsim.Sources.galaxy_catalog import GalaxyCatalog
from slsim.Sources.SupernovaeCatalog.supernovae_sample import (
    supernovae_host_galaxy_offset,
)

sn_type = "Ia"
band_list = ["i"]
lightcurve_time = np.linspace(-20, 100, 500)
absolute_mag_band = "bessellb"
mag_zpsys = "AB"
skypy_config = None
sky_area = Quantity(0.01, unit="deg2")
absolute_mag = None
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def test_supernovae_host_galaxy_offset():
    galaxy_catalog = GalaxyCatalog(
        cosmo=cosmo,
        skypy_config=skypy_config,
        sky_area=sky_area,
    )
    host_catalog = galaxy_catalog.galaxy_catalog()
    ra_off, dec_off = supernovae_host_galaxy_offset(host_catalog)

    ra_within_mean_radius = 0
    dec_within_mean_radius = 0
    mean_radius = np.rad2deg(np.mean(host_catalog["angular_size"]))

    for i in range(len(ra_off)):
        if np.abs(ra_off[i].value) <= mean_radius:
            ra_within_mean_radius += 1
        if np.abs(dec_off[i].value) <= mean_radius:
            dec_within_mean_radius += 1

    assert ra_within_mean_radius >= (2 / 3) * len(ra_off)
    assert dec_within_mean_radius >= (2 / 3) * len(dec_off)


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

    def test_supernovae_catalog(self):
        result = self.supernovae_catalog.supernovae_catalog()
        result2 = self.supernovae_catalog.supernovae_catalog(
            host_galaxy=False, lightcurve=False
        )
        assert "MJD" in result.colnames
        assert "z" in result.colnames
        assert "stellar_mass" in result.colnames
        assert len(result2.colnames) == 1


if __name__ == "__main__":
    pytest.main()
