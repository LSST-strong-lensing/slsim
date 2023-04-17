import numpy as np
import numpy.random as random
from sim_pipeline.selection import galaxy_cut


class Galaxies(object):
    """
    class describing early-type galaxies
    """
    def __init__(self, galaxy_list, kwargs_cut, cosmo):
        """

        :param galaxy_list: list of dictionary with galaxy parameters
        :type galaxy_list: astropy Table object
        :param kwargs_cut: cuts in parameters
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        """
        n = len(galaxy_list)
        # add missing keywords in astropy.Table object
        column_names = galaxy_list.colnames
        if 'e1' not in column_names or 'e2' not in column_names:
            galaxy_list['e1'] = -np.ones(n)
            galaxy_list['e2'] = -np.ones(n)
        if 'n_sersic' not in column_names:
            galaxy_list['n_sersic'] = -np.ones(n)
        # make cuts
        self._galaxy_select = galaxy_cut(galaxy_list, **kwargs_cut)

        self._num_select = len(self._galaxy_select)

    def draw_galaxy(self):

        index = random.randint(0, self._num_select - 1)
        galaxy = self._galaxy_select[index]

        if galaxy['e1'] == -1 or galaxy['e2'] == -1:
            e1, e2 = galaxy_projected_eccentricity(**galaxy)
            galaxy['e1'] = e1
            galaxy['e2'] = e2
        if galaxy['n_sersic'] == -1:
            galaxy['n_sersic'] = 1  # TODO make a better estimate with scatter
        return galaxy


def galaxy_projected_eccentricity(ellipticity, **kwargs):
    """
    projected eccentricity of early-type galaxies as a function of other deflector parameters

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1, e2 eccentricity components
    """
    e = ellipticity
    phi = np.random.uniform(0, np.pi)
    e1 = e * np.cos(phi)
    e2 = e * np.sin(phi)
    return e1, e2
