import numpy.random as random
from slsim.Sources.SourcePopulation.source_pop_base import SourcePopBase
from slsim.Lenses.selection import object_cut
from slsim.Sources.source import Source


class PointSources(SourcePopBase):
    """Class to describe point sources."""

    def __init__(
        self,
        point_source_list,
        cosmo,
        sky_area,
        kwargs_cut,
        list_type="astropy_table",
        point_source_type=None,
        point_source_kwargs={},
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
        :type kwargs_cut: dict
        :param point_source_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :param point_source_kwargs: dictionary of keyword arguments for a source. It should
         contain keywords for point_source_type and other keywords associated with
         pointsource. For supernova kwargs dict, please see documentation of
         SupernovaEvent class. For quasar kwargs dict, please see documentation of
         Quasar class.
        Eg of supernova kwargs: kwargs={"point_source_type": "supernova",
          "variability_model": "light_curve", "kwargs_variability": ["supernovae_lightcurve",
            "i", "r"], "sn_type": "Ia", "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab", "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": "/Users/narayankhadka/Downloads/sncosmo_sn_models/SALT3.NIR_WAVEEXT/"}.
         Other supported pointsource_types are "supernova", "quasar".
        """

        self.n = len(point_source_list)
        self._cosmo = cosmo
        self.sky_area = sky_area
        self.point_source_kwargs = point_source_kwargs
        self._point_source_type = point_source_type
        # make cuts
        self._point_source_select = object_cut(
            point_source_list, list_type=list_type, object_type="point", **kwargs_cut
        )

        self._num_select = len(self._point_source_select)
        super(SourcePopBase, self).__init__()
        self.source_type = "point_source"

    @property
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        number = self.n
        return number

    @property
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        return self._num_select

    def draw_source(self):
        """Choose source at random with the selected range.

        :return: dictionary of source
        """

        index = random.randint(0, self._num_select - 1)
        point_source = self._point_source_select[index]
        source_class = Source(
            cosmo=self._cosmo,
            point_source_type=self._point_source_type,
            **self.point_source_kwargs,
            **point_source
        )

        return source_class
