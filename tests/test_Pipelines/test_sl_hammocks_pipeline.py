import pytest
from astropy.cosmology import LambdaCDM
from astropy.cosmology import w0waCDM

from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline


class TestSkyPyPipeline(object):
    def setup_method(self):
        import os

        from astropy.units import Quantity

        self.sky_area = Quantity(value=0.001, unit="deg2")
        self.cosmo = LambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Ode0=0.7, Tcmb0=2.725)
        current_dir = os.path.dirname(__file__)
        self.slhammocks_config = os.path.join(
            current_dir,
            "../../data/SL-Hammocks",
            "pop_salp_a0001_zl001_to_5_wo_sub.csv",
        )
        self.pipeline = SLHammocksPipeline(
            slhammocks_config=self.slhammocks_config,
            sky_area=self.sky_area,
            cosmo=self.cosmo,
            z_min=0.01,
            z_max=5,
        )

    def test_specifying_skypy_configuration(self):
        # use case: Roman simulation

        import os
        import speclite
        from slsim.Pipelines import roman_speclite

        # load Roman speclite filters
        roman_speclite.configure_roman_filters()
        roman_filters = roman_speclite.filter_names()
        speclite.filters.load_filters(*roman_filters)

        skypy_config = os.path.join(
            os.path.dirname(__file__),
            "../../data/SkyPy/slhammock_skypy_roman.yml",
        )
        pipeline = SLHammocksPipeline(
            skypy_config=skypy_config,
            sky_area=self.sky_area,
            cosmo=self.cosmo,
            z_min=0.01,
            z_max=2,
        )
        assert pipeline._skypy_pipeline is not None

    def test_cosmology_initialization(self):
        sky_area = self.sky_area
        galaxy_cosmo0 = LambdaCDM(H0=70, Om0=0.15, Ob0=0.02, Ode0=0.85, Tcmb0=2.725)
        pipeline0 = SLHammocksPipeline(
            sky_area=sky_area, cosmo=galaxy_cosmo0, z_min=0.01, z_max=5
        )
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
        pipeline3 = SLHammocksPipeline(
            sky_area=sky_area, cosmo=cosmo3, z_min=0.01, z_max=5
        )
        halos_galaxies = pipeline3._pipeline
        # we expect the pipeline works:
        assert halos_galaxies[0]["z"] > 0

    def test_setting_too_wide_sky_area(self):
        from astropy.units import Quantity

        large_sky_area = Quantity(value=1000, unit="deg2")
        with pytest.raises(Exception) as excinfo:
            SLHammocksPipeline(
                slhammocks_config=self.slhammocks_config,
                sky_area=large_sky_area,
                cosmo=self.cosmo,
            )
        # Check the output
        assert "Now sky_area should be lower than" in str(
            excinfo.value
        ), "An exception with sky_area' message should be raised for too large sky_area"
