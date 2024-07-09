import pytest
import numpy as np
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.galaxy_catalog import GalaxyCatalog
from slsim.Sources.supernovae_host_match import SupernovaeHostMatch


skypy_config = None
sky_area = Quantity(0.001, unit="deg2")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

supernovae_catalog = np.linspace(0, 2.379, 20)
galaxy_catalog_class = GalaxyCatalog(
    cosmo=cosmo,
    skypy_config=skypy_config,
    sky_area=sky_area,
)
galaxy_catalog = galaxy_catalog_class.galaxy_catalog()


class TestSupernovaeHostMatch:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.match = SupernovaeHostMatch(
            supernovae_catalog=supernovae_catalog, galaxy_catalog=galaxy_catalog
        )

    def test_match(self):
        result = self.match.match()
        assert len(result["stellar_mass"]) == 20
        assert supernovae_catalog[5] <= (result["z"][5] + 0.1)
        assert supernovae_catalog[5] >= (result["z"][5] - 0.1)


if __name__ == "__main__":
    pytest.main()
