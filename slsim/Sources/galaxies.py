import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Util import param_util
from slsim.Sources.source_pop_base import SourcePopBase
from astropy.table import Column


class Galaxies(SourcePopBase):
    """Class describing elliptical galaxies."""

    def __init__(
        self, galaxy_list, kwargs_cut, cosmo, sky_area, list_type="astropy_table"
    ):
        """

        :param galaxy_list: list of dictionary with galaxy parameters
        :type galaxy_list: astropy Table object
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :param list_type: format of the source catalog file. Currently, it supports a single
        astropy table or a list of astropy tables.
        :type sky_area: `~astropy.units.Quantity`
        """
        super(Galaxies, self).__init__(cosmo=cosmo, sky_area=sky_area)
        self.n = len(galaxy_list)
        # add missing keywords in astropy.Table object
        if list_type == "astropy_table":
            column_names = galaxy_list.colnames
            if "ellipticity" not in column_names:
                raise ValueError("required parameters missing in galaxy_list" "columns")
            if "e1" not in column_names or "e2" not in column_names:
                galaxy_list["e1"] = -np.ones(self.n)
                galaxy_list["e2"] = -np.ones(self.n)
            if "n_sersic" not in column_names:
                galaxy_list["n_sersic"] = -np.ones(self.n)
        else:
            column_names = galaxy_list[0].colnames
            if "ellipticity" not in column_names:
                raise ValueError("required parameters missing in galaxy_list columns")
            if "e1" not in column_names or "e2" not in column_names:
                for table in galaxy_list:
                    new_column_length = len(table)
                    new_column_1 = Column([-1.0] * new_column_length, name="e1")
                    new_column_2 = Column([-1.0] * new_column_length, name="e2")
                    table.add_columns([new_column_1, new_column_2])
            if "n_sersic" not in column_names:
                for table in galaxy_list:
                    new_column_length = len(table)
                    new_column = Column([-1] * new_column_length, name="n_sersic")
                    table.add_column(new_column)
        # make cuts
        self._galaxy_select = deflector_cut(
            galaxy_list, list_type=list_type, **kwargs_cut
        )

        self._num_select = len(self._galaxy_select)
        self.list_type = list_type

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
        galaxy = self._galaxy_select[index]

        if galaxy["e1"] == -1 or galaxy["e2"] == -1:
            e1, e2 = galaxy_projected_eccentricity(float(galaxy["ellipticity"]))
            galaxy["e1"] = e1
            galaxy["e2"] = e2
        if galaxy["n_sersic"] == -1:
            galaxy["n_sersic"] = 1  # TODO make a better estimate with scatter
        return galaxy


def galaxy_projected_eccentricity(ellipticity):
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :return: e1, e2 eccentricity components
    """
    e = param_util.epsilon2e(ellipticity)
    phi = np.random.uniform(0, np.pi)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2
