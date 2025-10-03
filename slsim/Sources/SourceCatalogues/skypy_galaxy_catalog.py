from slsim.Pipelines.skypy_pipeline import SkyPyPipeline

"""References:
Wang et al. 2013
"""


class GalaxyCatalog(object):
    """Class to generate a galaxy catalog."""

    def __init__(self, cosmo, skypy_config, sky_area):
        """

        :param cosmo: astropy.cosmology instance
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self.cosmo = cosmo
        self.skypy_config = skypy_config
        self.sky_area = sky_area

    def galaxy_catalog(self):
        """Generates galaxy catalog and those galaxies can be used as
        supernovae host galaxies.

        :return: supernovae host galaxy candidate catalog
        :return type: astropy Table
        """
        pipeline = SkyPyPipeline(
            skypy_config=self.skypy_config,
            sky_area=self.sky_area,
            filters=None,
            cosmo=self.cosmo,
        )
        galaxy_table = pipeline.blue_galaxies
        return galaxy_table
