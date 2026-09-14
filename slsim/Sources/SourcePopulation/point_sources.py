from slsim.Sources.SourcePopulation.source_pop_base import SourcePopBase
from slsim.Sources.source import Source


class PointSources(SourcePopBase):
    """Class to describe point sources."""

    def __init__(
        self,
        point_source_list,
        cosmo,
        sky_area,
        kwargs_cut=None,
        point_source_type=None,
        joint_point_source_kwargs=None,
    ):
        """

        :param point_source_list: list of dictionary with quasar parameters or astropy
         table.
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max. These are
         the arguments that go into the deflector_cut() definition which is a general
         definition for performing given cuts in given catalog. For the supernovae
         sample, we can only apply redshift cuts because supernovae sample contains only
         redshift in this stage.
        :type kwargs_cut: dict or None
        :param point_source_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :param joint_point_source_kwargs: dictionary of keyword arguments for point sources that are joint. It should
         contain keywords for point_source_type and other keywords associated with
         point source. Provides population-level default values applied uniformly
         to every draw. Any key here may be overridden on a per-object basis by
         including a same-named column in `point_source_list` -- the catalog value
         takes precedence.
        """
        if joint_point_source_kwargs is None:
            joint_point_source_kwargs = {}
        self._joint_point_source_kwargs = joint_point_source_kwargs
        # make cuts
        if kwargs_cut is None:
            kwargs_cut = {}
        if "object_type" not in kwargs_cut:
            # make sure the magnitude selection is on the point source and not the extended one
            kwargs_cut["object_type"] = "point"

        super().__init__(
            object_list=point_source_list,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut=kwargs_cut,
            point_source_type=point_source_type,
        )

    def draw_source(self, z_max=None, z_min=None, object_index=None):
        """Choose source at random within the selected redshift range.

        :param z_max: maximum redshift limit for the point source to be
            drawn. If no point source is found for this limit, None will
            be returned.
        :param z_min: minimum redshift limit for the point source to be
            drawn. If no point source is found for this limit, None will
            be returned.
        :param object_index: index of point source to pick (if provided)
        :return: instance of Source class
        """
        point_source = self.draw_object(
            z_max=z_max, z_min=z_min, galaxy_index=object_index
        )

        # per-object catalog values override the joint/population-level defaults
        # on key collision, rather than raising (as a direct double-** unpack would)
        merged_kwargs = {**self._joint_point_source_kwargs, **point_source}
        source_class = Source(
            cosmo=self._cosmo,
            point_source_type=self._point_source_type,
            **merged_kwargs
        )

        return source_class
