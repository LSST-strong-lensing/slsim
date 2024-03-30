import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Deflectors.velocity_dispersion import vel_disp_sdss
from slsim.Util import param_util
from slsim.Deflectors.deflectors_base import DeflectorsBase


class GalaxyClusterLenses(DeflectorsBase):
    """Class describing a population of cluster-scale halos + subhaloes."""

    def __init__(self, galaxy_list, kwargs_cut, kwargs_mass2light, cosmo, sky_area):
        """

        :param galaxy_list: list of dictionary with halo parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        """
        super().__init__(
            deflector_table=galaxy_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        ...
        self._num_select = 0

    def deflector_number(self):
        """

        :return: number of deflectors
        """
        number = self._num_select
        return number

    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """

        pass


def halo_projected_eccentricity(ellipticity, **kwargs):
    """Projected eccentricity of halo as a function of other deflector
    parameters.

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1_light, e2_light,e1_mass, e2_mass eccentricity components
    """
    pass


