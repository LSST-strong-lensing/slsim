import pytest
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.galaxy_catalog import GalaxyCatalog
from slsim.Sources.galaxy_catalog import supernovae_host_galaxy_offset
import numpy as np

skypy_config = None
sky_area = Quantity(0.001, unit="deg2")
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


class TestGalaxyCatalog:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.galaxy_catalog = GalaxyCatalog(
            cosmo=cosmo,
            skypy_config=skypy_config,
            sky_area=sky_area,
        )

    def test_host_galaxy_catalog(self):
        result = self.galaxy_catalog.galaxy_catalog()
        assert all(result["z"] <= 2.379)


if __name__ == "__main__":
    pytest.main()
