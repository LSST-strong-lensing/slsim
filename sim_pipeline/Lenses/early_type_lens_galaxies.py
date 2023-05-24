import numpy as np
import numpy.random as random
from sim_pipeline.selection import galaxy_cut
from sim_pipeline.Lenses.velocity_dispersion import vel_disp_sdss


class EarlyTypeLensGalaxies(object):
    """
    class describing early-type galaxies
    """
    def __init__(self, galaxy_list, kwargs_cut, kwargs_mass2light, cosmo, sky_area):
        """

        :param galaxy_list: list of dictionary with galaxy parameters of early-type galaxies
         (currently supporting skypy pipelines)
        :param kwargs_cut: cuts in parameters
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of solid angle.
        """
        self.n = len(galaxy_list)
        column_names = galaxy_list.colnames
        if 'vel_disp' not in column_names:
            galaxy_list['vel_disp'] = -np.ones(self.n)
        if 'e1_light' not in column_names or 'e2_light' not in column_names:
            galaxy_list['e1_light'] = -np.ones(self.n)
            galaxy_list['e2_light'] = -np.ones(self.n)
        if 'e1_mass' not in column_names or 'e2_mass' not in column_names:
            galaxy_list['e1_mass'] = -np.ones(self.n)
            galaxy_list['e2_mass'] = -np.ones(self.n)
        if 'n_sersic' not in column_names:
            galaxy_list['n_sersic'] = -np.ones(self.n)

        self._galaxy_select = galaxy_cut(galaxy_list, **kwargs_cut)
        self._num_select = len(self._galaxy_select)

        z_min, z_max = 0, np.max(self._galaxy_select['z'])
        redshift = np.linspace(start=z_min, stop=z_max, num=20)
        z_list, vel_disp_list = vel_disp_sdss(sky_area, redshift, vd_min=100, vd_max=500, cosmology=cosmo, noise=True)
        # sort for stellar masses
        self._galaxy_select.sort('stellar_mass')
        # sort velocity dispersion
        vel_disp_list = np.sort(vel_disp_list)
        num_vel_disp = len(vel_disp_list)
        # print(num_vel_disp, self._num_select, z_max, 'test ')
        if num_vel_disp > self._num_select:
            # randomly select
            pass
            # np.random.choice()
        # TODO: abundance match velocity dispersion with early-type galaxy catalogue

        # TODO: random reshuffle of matched list

    def deflector_number(self):
        number = self.n
        return number

    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """

        index = random.randint(0, self._num_select - 1)
        deflector = self._galaxy_select[index]
        if deflector['vel_disp'] == -1:
            stellar_mass = deflector['stellar_mass']
            vel_disp = vel_disp_from_m_star(stellar_mass)
            deflector['vel_disp'] = vel_disp
        if deflector['e1_light'] == -1 or deflector['e2_light'] == - 1:
            e1_light, e2_light, e1_mass, e2_mass = early_type_projected_eccentricity(**deflector)
            deflector['e1_light'] = e1_light
            deflector['e2_light'] = e2_light
            deflector['e1_mass'] = e1_mass
            deflector['e2_mass'] = e2_mass
        if deflector['n_sersic'] == -1:
            deflector['n_sersic'] = 4  # TODO make a better estimate with scatter
        return deflector


def early_type_projected_eccentricity(ellipticity, **kwargs):
    """
    projected eccentricity of early-type galaxies as a function of other deflector parameters

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1_light, e2_light,e1_mass, e2_mass eccentricity components
    """
    e_light = ellipticity
    phi_light = np.random.uniform(0, np.pi)
    e1_light = e_light * np.cos(phi_light)
    e2_light = e_light * np.sin(phi_light)
    e_mass = 0.5 * ellipticity + np.random.normal(loc=0, scale=0.1)
    phi_mass = phi_light + np.random.normal(loc=0, scale=0.1)
    e1_mass = e_mass * np.cos(phi_mass)
    e2_mass = e_mass * np.sin(phi_mass)
    return e1_light, e2_light, e1_mass, e2_mass


def vel_disp_from_m_star(m_star):
    """
    function for calculate the velocity dispersion from the staller mass using empirical relation for
    early type galaxies

    The power-law formula is given by:

    .. math::

         V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11} M_\\odot} \\right)^{0.24}

    2.32,0.24 is the parameters from [1] table 2
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and total mass correlations of massive
    early-type galaxies." The Astrophysical Journal 724.1 (2010): 511.

    :param m_star: stellar mass in the unit of solar mass
    :return: the velocity dispersion ("km/s")

    """
    v_disp = (np.power(10, 2.32) * np.power(m_star/1e11, 0.24))
    return v_disp
