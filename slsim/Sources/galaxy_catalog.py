from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
import numpy as np
from scipy import stats
import astropy.units as units
from astropy.coordinates import SkyCoord

"""References:
Wang et al. 2013
"""


def supernovae_host_galaxy_offset(host_galaxy_catalog):
    """This function generates random supernovae offsets from their host galaxy center
    based on observed data. (Wang et al. 2013)

    :param host_galaxy_catalog: catalog of host galaxies matched with supernovae (must
        have 'angular_size' column)
    :type host_galaxy_catalog: astropy Table
    :return: random ra_off and dec_off for each supernovae
    """
    # Select offset ratios based on observed offset distribution (Wang et al. 2013)
    offset_ratios = list(
        stats.lognorm.rvs(
            0.764609, loc=-0.0284546, scale=0.450885, size=len(host_galaxy_catalog)
        )
    )

    offsets = []
    position_angle = []

    for i in range(len(host_galaxy_catalog)):
        while offset_ratios[i] > 3:
            offset_ratios[i] = stats.lognorm.rvs(
                0.764609, loc=-0.0284546, scale=0.450885, size=1
            )[0]

        offsets.append(offset_ratios[i] * list(host_galaxy_catalog["angular_size"])[i])
        position_angle.append(np.random.uniform(0, 360))

    host_center = SkyCoord(1 * units.deg, 1 * units.deg, frame="icrs")
    offsets = host_center.directional_offset_by(position_angle, offsets)

    ra_off = offsets.ra
    ra_off = ra_off - 1 * units.deg
    dec_off = offsets.dec
    dec_off = dec_off - 1 * units.deg

    return ra_off, dec_off


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
