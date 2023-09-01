from sim_pipeline.Deflectors.all_lens_galaxies import AllLensGalaxies
class test_all_galaxies(object):
        def setup_method(self): 
            sky_area = Quantity(value=0.1, unit='deg2')
            pipeline = SkyPyPipeline(skypy_config=skypy_config, sky_area=sky_area, filters=filters)
            kwargs_deflector_cut = {}
            kwargs_mass2light = {}
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            self.lens_galaxies = AllLensGalaxies(pipeline.red_galaxies, pipeline.blue_galaxies,
                                                    kwargs_cut=kwargs_deflector_cut, kwargs_mass2light=kwargs_mass2light,
                                                    cosmo=cosmo, sky_area=sky_area)
            


if __name__ == '__main__':
    pytest.main()
