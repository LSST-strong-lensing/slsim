from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline
from astropy.cosmology import LambdaCDM
import pytest


class TestSkyPyPipeline(object):
    def setup_method(self):
        from astropy.units import Quantity

        self.sky_area = Quantity(value=0.001, unit="deg2")
        self.cosmo = LambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Ode0=0.7, Tcmb0=2.725)
        slhammocks_config = "../../data/SL-Hammocks/gal_pop_Salpeter_10deg2_zl2.csv"
        self.pipeline = SLHammocksPipeline(
            slhammocks_config=slhammocks_config,
            sky_area=self.sky_area,
            cosmo=self.cosmo,
        )

    def test_cosmology_initialization(self):

        sky_area = self.sky_area
        galaxy_cosmo0 = LambdaCDM(H0=70, Om0=0.15, Ob0=0.02, Ode0=0.85, Tcmb0=2.725)
        pipeline0 = SLHammocksPipeline(sky_area=sky_area, cosmo=galaxy_cosmo0)
        halos_galaxies = pipeline0._pipeline
        cosmo0_halo_number = len(halos_galaxies)

        # LambdaCDM model
        halos_galaxies = self.pipeline._pipeline
        # we expect the pipeline works:
        assert halos_galaxies[0]["z"] > 0

        # we expect the halo number from this cosmo is
        # larger than the previous cosmo, indicating change of cosmology:
        assert len(halos_galaxies) > cosmo0_halo_number

        # check w0waCDM model
        cosmo3 = w0waCDM(
            H0=70, Om0=0.3, Ob0=0.05, Ode0=0.7, w0=-1.0, wa=0.0, Tcmb0=2.725
        )
        pipeline3 = SLHammocksPipeline(sky_area=sky_area, cosmo=cosmo3)
        halos_galaxies = pipeline3._pipeline
        # we expect the pipeline works:
        assert halos_galaxies[0]["z"] > 0

    def test_setting_too_wide_sky_area(self):
        from astropy.units import Quantity

        slhammocks_config = "../../data/SL-Hammocks/gal_pop_Salpeter_10deg2_zl2.csv"
        large_sky_area = Quantity(value=1000, unit="deg2")
        with pytest.raises(Exception) as excinfo:
            SLHammocksPipeline(
                slhammocks_config=slhammocks_config,
                sky_area=large_sky_area,
                cosmo=self.cosmo,
            )
        # Check the output
        assert "Now sky_area should be lower than" in str(
            excinfo.value
        ), "An exception with sky_area' message should be raised for too large sky_area"

    def test_ellip_from_axis_ratio2epsilon(self):
        # Translates ellipticity definitions from.
        # math::
        #    ellipticity = \\equic \\1 - q
        # where q is axis ratio to ellipticity in SkyPyPipeline
        #    epsilon = \\equic \\frac{1 - q^2}{1 + q^2}
        from slsim.Pipelines.sl_hammocks_pipeline import ellip_from_axis_ratio2epsilon

        assert ellip_from_axis_ratio2epsilon(0.5) == 0.6
