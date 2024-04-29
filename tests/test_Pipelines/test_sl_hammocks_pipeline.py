from slsim.Pipelines.sl_hammocks_pipeline import SLHammocksPipeline
from astropy.cosmology import LambdaCDM, FlatwCDM, w0waCDM


class TestSkyPyPipeline(object):
    def setup_method(self):
        from astropy.units import Quantity

        sky_area = Quantity(value=0.05, unit="deg2")
        cosmo = LambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Ode0=0.7, Tcmb0=2.725)
        self.pipeline = SLHammocksPipeline(sky_area=sky_area, cosmo=cosmo)

    def test_cosmology_initialization(self):
        from astropy.units import Quantity

        sky_area = Quantity(value=0.05, unit="deg2")
        galaxy_cosmo0 = LambdaCDM(H0=70, Om0=0.15, Ob0=0.02, Ode0=0.85, Tcmb0=2.725)
        pipeline0 = SLHammocksPipeline(sky_area=sky_area, cosmo=galaxy_cosmo0)
        halos_galaxies = pipeline0._pipeline
        cosmo0_halo_number = len(halos_galaxies)

        # LambdaCDM model
        cosmo1 = LambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Ode0=0.7, Tcmb0=2.725)
        pipeline1 = SLHammocksPipeline(sky_area=sky_area, cosmo=cosmo1)
        halos_galaxies = pipeline1._pipeline
        # we expect the pipeline works:
        assert halos_galaxies[0]["z"] > 0

        # we expect the halo number from this cosmo is
        # larger than the previous cosmo, indicating change of cosmology:
        assert len(halos_galaxies) > cosmo0_halo_number

        # check FlatwCDM model
        cosmo3 = FlatwCDM(H0=70, Om0=0.3, Ob0=0.05, w0=-1.0, Tcmb0=2.725)
        pipeline3 = SLHammocksPipeline(sky_area=sky_area, cosmo=cosmo3)
        halos_galaxies = pipeline3._pipeline
        # we expect the pipeline works:
        assert halos_galaxies[0]["z"] > 0

        # check w0waCDM model
        cosmo4 = w0waCDM(
            H0=70, Om0=0.3, Ob0=0.05, Ode0=0.7, w0=-1.0, wa=0.0, Tcmb0=2.725
        )
        pipeline4 = SLHammocksPipeline(sky_area=sky_area, cosmo=cosmo4)
        halos_galaxies = pipeline4._pipeline
        assert halos_galaxies[0]["z"] > 0

    def test_ellip_from_axis_ratio2epsilon(self):
        # Translates ellipticity definitions from.
        # math::
        #    ellipticity = \\equic \\1 - q
        # where q is axis ratio to ellipticity in SkyPyPipeline
        #    epsilon = \\equic \\frac{1 - q^2}{1 + q^2}
        from slsim.Pipelines.sl_hammocks_pipeline import ellip_from_axis_ratio2epsilon

        assert ellip_from_axis_ratio2epsilon(0.5) == 0.6
