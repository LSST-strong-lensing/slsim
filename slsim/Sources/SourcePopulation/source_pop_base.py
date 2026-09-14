import numpy.random as random
from slsim.Lenses.selection import object_cut


class SourcePopBase(object):
    """Base class with functions all source classes must have to be able to
    render populations."""

    def __init__(
        self,
        object_list,
        cosmo,
        sky_area,
        kwargs_cut=None,
        point_source_type=None,
        extended_source_type=None,
    ):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param point_source_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :param extended_source_type: keyword for number of sersic profile to use in source
         light model. accepted kewords: "single_sersic", "double_sersic".
        """
        self.sky_area = sky_area
        self._cosmo = cosmo
        # These quantities are defined here because Source class these quantities and
        # None act as default values.
        self._point_source_type = point_source_type
        self._extended_source_type = extended_source_type

        if kwargs_cut is None:
            kwargs_cut = {}
        self._objects_select = object_cut(object_list, **kwargs_cut)
        self._num_select = len(self._objects_select)
        self._full_object_list = object_list
        self._object_number = len(object_list)

    @property
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        return self._object_number

    @property
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        return self._num_select

    def draw_object(self, z_max=None, z_min=None, galaxy_index=None):
        """Chose object from catalog at random.

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param galaxy_index: index of galaxy to pic (if provided)
        :return: dictionary of source in the form of the original
            catalog
        """
        if galaxy_index is not None:
            object = self._full_object_list[galaxy_index]

        elif z_max is not None or z_min is not None:
            if z_max is None:
                z_max = 1100
            if z_min is None:
                z_min = 0
            filtered_galaxies = self._objects_select[
                (self._objects_select["z"] < z_max)
                & (z_min < self._objects_select["z"])
            ]
            if len(filtered_galaxies) == 0:
                return None
            else:
                index = random.randint(0, len(filtered_galaxies))
                object = filtered_galaxies[index]
        else:
            index = random.randint(0, self._num_select)
            object = self._objects_select[index]
        return object
