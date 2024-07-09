from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
import numpy as np


class GalaxyCatalog(object):
    """Class to generate a galaxy catalog."""

    def __init__(
        self,
        cosmo,
        skypy_config,
        sky_area
    ):
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
        """Generates galaxy catalog and those galaxies can be used as supernovae host
        galaxies.

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
        galaxy_table_cut = galaxy_table[galaxy_table["z"] <= 2.379]
        return galaxy_table_cut

    def supernovae_host_galaxy_offset(self, supernovae_number):
        """This function generates random supernovae offsets from their host galaxy
        center.

        # TODO: use supernovae and host galaxy parameters to compute more realistic
        offset.

        :param supernovae_number: number of supernovae
        :return: random ra_off and dec_off for each supernovae.
        """
        # Limits used here are mostly arbitrary. More realistic supernovae-host galaxy
        # offset is needed.
        ra_off = np.random.uniform(-5, 5, supernovae_number)
        dec_off = np.random.uniform(-5, 5, supernovae_number)
        return ra_off, dec_off
