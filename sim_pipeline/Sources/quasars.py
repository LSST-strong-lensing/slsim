from sim_pipeline.Sources.source_base import SourceBase


class Quasars(SourceBase):
    """Class to describe quasars as sources."""

    def __init__(self, cosmo, sky_area):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        super(Quasars, self).__init__(cosmo=cosmo, sky_area=sky_area)
