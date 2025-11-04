from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.cosmology import (
    LambdaCDM,
    FlatLambdaCDM,
    FlatwCDM,
    w0waCDM,
    default_cosmology,
)
import os


class TestSkyPyPipeline(object):
    def setup_method(self):
        from astropy.units import Quantity

        cosmo_test = FlatLambdaCDM(H0=70, Om0=0.3)
        self.sky_area = Quantity(value=0.001, unit="deg2")
        self.pipeline = SkyPyPipeline(
            skypy_config=None,
            sky_area=self.sky_area,
            z_min=0,
            z_max=4.09,
            filters=["g", "r", "i", "z", "y", "u"],
            cosmo=cosmo_test,
        )
        self.pipeline2 = SkyPyPipeline(
            skypy_config="lsst_like_old", sky_area=self.sky_area
        )

        path = os.path.dirname(__file__)
        new_path = path.replace("test_Pipelines", "TestData/")
        self.pipeline3 = SkyPyPipeline(
            skypy_config=new_path + "lsst_like_test_1.yml", sky_area=self.sky_area
        )

    def test_default_pipeline(self):
        pipeline_default = SkyPyPipeline(sky_area=self.sky_area)
        blue_galaxies = pipeline_default.blue_galaxies
        assert blue_galaxies[0]["z"] > 0

    def test_cosmology_initialization(self):
        galaxy_cosmo0 = LambdaCDM(H0=70, Om0=0.01, Ode0=0.99)
        pipeline0 = SkyPyPipeline(cosmo=galaxy_cosmo0, sky_area=self.sky_area)
        blue_galaxies = pipeline0.blue_galaxies
        red_galaxies = pipeline0.red_galaxies
        cosmo0_blue_number = len(blue_galaxies)
        cosmo0_red_number = len(red_galaxies)

        # LambdaCDM model
        cosmo1 = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
        pipeline1 = SkyPyPipeline(cosmo=cosmo1, sky_area=self.sky_area)
        blue_galaxies = pipeline1.blue_galaxies
        red_galaxies = pipeline1.red_galaxies
        # we expect the pipeline works:
        assert blue_galaxies[0]["z"] > 0
        assert red_galaxies[0]["z"] > 0

        # we expect the galaxy number from this cosmo is significantly
        # less than the previous cosmo, indicating change of cosmology:
        assert len(blue_galaxies) * 2 < cosmo0_blue_number
        assert len(red_galaxies) * 2 < cosmo0_red_number

        # check FlatwCDM model
        cosmo3 = FlatwCDM(H0=70, Om0=0.3, w0=-1.0)
        pipeline3 = SkyPyPipeline(cosmo=cosmo3, sky_area=self.sky_area)
        red_galaxies = pipeline3.red_galaxies
        assert red_galaxies[0]["z"] > 0

        # check w0waCDM model
        cosmo4 = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-1.0, wa=0.0)
        pipeline4 = SkyPyPipeline(cosmo=cosmo4, sky_area=self.sky_area)
        blue_galaxies = pipeline4.blue_galaxies
        assert blue_galaxies[0]["z"] > 0

        galaxy_cosmo5 = default_cosmology.get()
        pipeline5 = SkyPyPipeline(cosmo=galaxy_cosmo5, sky_area=self.sky_area)
        blue_galaxies5 = pipeline5.blue_galaxies
        assert blue_galaxies5[0]["z"] > 0

    def test_blue_galaxies(self):
        blue_galaxies = self.pipeline.blue_galaxies
        assert blue_galaxies[0]["z"] > 0

    def test_red_galaxies(self):
        red_galaxies = self.pipeline.red_galaxies
        assert red_galaxies[0]["z"] > 0
        assert len(self.pipeline2.red_galaxies["z"]) > 0
        assert len(self.pipeline3.red_galaxies["z"]) > 0

    def test_redshift_range(self):
        blue_galaxies = self.pipeline.blue_galaxies
        red_galaxies = self.pipeline.red_galaxies
        assert max(blue_galaxies["z"]) <= 4.09
        assert min(blue_galaxies["z"]) >= 0.0
        assert max(red_galaxies["z"]) <= 4.09
        assert min(red_galaxies["z"]) >= 0.0

    def test_filters(self):
        blue_galaxies = self.pipeline.blue_galaxies
        red_galaxies = self.pipeline.red_galaxies
        assert "mag_g" in blue_galaxies.colnames
        assert "mag_r" in blue_galaxies.colnames
        assert "mag_i" in blue_galaxies.colnames
        assert "mag_z" in blue_galaxies.colnames
        assert "mag_y" in blue_galaxies.colnames
        assert "mag_u" in blue_galaxies.colnames

        assert "mag_g" in red_galaxies.colnames
        assert "mag_r" in red_galaxies.colnames
        assert "mag_i" in red_galaxies.colnames
        assert "mag_z" in red_galaxies.colnames
        assert "mag_y" in red_galaxies.colnames
        assert "mag_u" in red_galaxies.colnames
