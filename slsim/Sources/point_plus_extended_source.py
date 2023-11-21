import numpy.random as random
from slsim.Sources.source_pop_base import SourcePopBase
from slsim.Util.param_util import epsilon2e
import numpy as np
from slsim.selection import deflector_cut
from slsim.Sources.galaxies import galaxy_projected_eccentricity


class PESource(SourcePopBase):
    """Class to describe point and extended sources."""

    def __init__(
        self,
        pes_list,
        cosmo,
        sky_area,
        kwargs_cut,
        variability_model=None,
        kwargs_variability_model=None,
    ):
        """

        :param pes_list: list of dictionary with point and extended source parameters
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         source.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual source.
        """
        super(PESource, self).__init__(
            cosmo=cosmo,
            sky_area=sky_area,
            variability_model=variability_model,
            kwargs_variability_model=kwargs_variability_model,
        )
        self.n = len(pes_list)
        ## check missing kewords in astropy table.
        column_names = pes_list.colnames
        if "ellipticity" not in column_names:
            raise ValueError("required parameters missing in galaxy_list columns")
        if "e1" not in column_names or "e2" not in column_names:
            pes_list["e1"] = -np.ones(self.n)
            pes_list["e2"] = -np.ones(self.n)
        if "n_sersic" not in column_names:
            pes_list["n_sersic"] = -np.ones(self.n)

        self._point_extended_select = deflector_cut(pes_list, **kwargs_cut)

        self._num_select = len(self._point_extended_select)

    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        number = self.n
        return number

    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """

        index = random.randint(0, self._num_select - 1)
        point_extended_source = self._point_extended_select[index]

        if point_extended_source["e1"] == -1 or point_extended_source["e2"] == -1:
            e1, e2 = galaxy_projected_eccentricity(
                float(point_extended_source["ellipticity"])
            )
            point_extended_source["e1"] = e1
            point_extended_source["e2"] = e2
        if point_extended_source["n_sersic"] == -1:
            point_extended_source[
                "n_sersic"
            ] = 1  # TODO make a better estimate with scatter
        return point_extended_source
