import pytest
import numpy as np
from astropy.units import Quantity
from astropy import units
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SourceCatalogues.SupernovaeCatalog import SupernovaeCatalog
from slsim.Sources.SourceCatalogues.skypy_galaxy_catalog import GalaxyCatalog
from slsim.Sources.SourceCatalogues.SupernovaeCatalog.supernovae_sample import (
    supernovae_host_galaxy_offset,
)
import slsim.Pipelines as pipelines

sn_type = "Ia"
band_list = ["i"]
lightcurve_time = np.linspace(-20, 50, 100)
absolute_mag_band = "bessellb"
mag_zpsys = "AB"
skypy_config = None
sky_area = Quantity(0.05, unit="deg2")
absolute_mag = None
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def test_supernovae_host_galaxy_offset():
    galaxy_catalog = GalaxyCatalog(
        cosmo=cosmo,
        skypy_config=skypy_config,
        sky_area=Quantity(0.001, unit="deg2"),
    )
    host_catalog = galaxy_catalog.galaxy_catalog()
    x_off, y_off, e1, e2 = supernovae_host_galaxy_offset(host_catalog)

    x_within_mean_radius = 0
    y_within_mean_radius = 0
    mean_radius = np.rad2deg(np.mean(host_catalog["angular_size"])) * units.deg
    mean_radius = mean_radius.to(units.arcsec)

    for i in range(len(x_off)):

        if np.abs(x_off[i]) <= mean_radius.value:
            x_within_mean_radius += 1
        if np.abs(y_off[i]) <= mean_radius.value:
            y_within_mean_radius += 1
        e1[i] = abs(e1[i])
        e2[i] = abs(e2[i])

    assert x_within_mean_radius >= (2 / 3) * len(x_off)
    assert y_within_mean_radius >= (2 / 3) * len(y_off)
    assert min(np.abs(e1)) > 0
    assert min(np.abs(e2)) > 0
    assert max(np.abs(e1)) < 1
    assert max(np.abs(e2)) < 1


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
        # generate galaxy population using skypy pipeline.
        galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
            skypy_config=skypy_config, sky_area=sky_area, filters=None, cosmo=cosmo
        )
        blue_galaxie = galaxy_simulation_pipeline.blue_galaxies
        self.supernovae_catalog2 = SupernovaeCatalog(
            sn_type=sn_type,
            band_list=band_list,
            lightcurve_time=lightcurve_time,
            absolute_mag_band=absolute_mag_band,
            mag_zpsys=mag_zpsys,
            cosmo=cosmo,
            skypy_config=skypy_config,
            sky_area=sky_area,
            absolute_mag=absolute_mag,
            host_galaxy_candidate=blue_galaxie,
        )

    def test_supernovae_catalog(self):
        result = self.supernovae_catalog.supernovae_catalog()
        result2 = self.supernovae_catalog.supernovae_catalog(
            host_galaxy=False, lightcurve=False
        )
        result3 = self.supernovae_catalog2.supernovae_catalog(
            host_galaxy=True, lightcurve=False
        )
        assert "MJD" in result.colnames
        assert "z" in result.colnames
        assert "stellar_mass" in result.colnames
        assert "e1" in result.colnames
        assert len(result2.colnames) == 1
        assert "x_off" in result3.colnames
        assert "y_off" in result3.colnames
        assert self.supernovae_catalog2.host_galaxy_candidate is not None


if __name__ == "__main__":
    pytest.main()
