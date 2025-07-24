import pytest
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SourceCatalogues.skypy_galaxy_catalog import GalaxyCatalog

skypy_config = None
sky_area = Quantity(0.001, unit="deg2")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


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
        assert all(result["z"] <= 5.01)


if __name__ == "__main__":
    pytest.main()
