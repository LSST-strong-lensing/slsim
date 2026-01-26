import pytest
import numpy as np
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SourceCatalogues.skypy_galaxy_catalog import GalaxyCatalog
from slsim.Sources.Supernovae.supernovae_host_match import SupernovaeHostMatch

skypy_config = None
sky_area = Quantity(0.1, unit="deg2")
sky_area2 = Quantity(1, unit="deg2")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

supernovae_catalog = np.linspace(0, 2.379, 20)
galaxy_catalog_class = GalaxyCatalog(
    cosmo=cosmo,
    skypy_config=skypy_config,
    sky_area=sky_area,
)
galaxy_catalog = galaxy_catalog_class.galaxy_catalog()

galaxy_catalog_class2 = GalaxyCatalog(
    cosmo=cosmo,
    skypy_config=skypy_config,
    sky_area=sky_area2,
)
galaxy_catalog2 = galaxy_catalog_class2.galaxy_catalog()


class TestSupernovaeHostMatch:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.match = SupernovaeHostMatch(
            supernovae_catalog=supernovae_catalog, galaxy_catalog=galaxy_catalog
        )
        self.match2 = SupernovaeHostMatch(
            supernovae_catalog=supernovae_catalog, galaxy_catalog=galaxy_catalog2
        )

    def test_match(self):
        result = self.match.match()
        result2 = self.match2.match()
        assert len(result["stellar_mass"]) == 20
        assert supernovae_catalog[5] <= (result["z"][5] + 0.1)
        assert supernovae_catalog[5] >= (result["z"][5] - 0.1)
        assert supernovae_catalog[5] <= (result2["z"][5] + 0.05)
        assert supernovae_catalog[5] >= (result2["z"][5] - 0.05)


if __name__ == "__main__":
    pytest.main()
