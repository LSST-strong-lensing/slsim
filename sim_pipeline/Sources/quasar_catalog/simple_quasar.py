import numpy as np
from astropy.table import Table

def quasar_catalog(**kwargs_quasars):
    """ Creates an simple catalog of quasars.

    :param kwargs_quasars: a dict of the form kwargs_quasars = {'number': 50000, 
     'z_min': 0.1, 'z_max': 5, 'm_min': 17, 'm_max': 25} and param are explained below.
    :param number: number of sources we want
    :param z_min: minimum redshift for sources
    :param z_max: maximum redshift for sources
    :param m_min: minimum magnitude for sources in r band
    :param m_max: maximum magnitude for sources in r band
    :return: an astropy table of quasar catalog
    """
    num_quasars = kwargs_quasars.get('number', 50000)
    z_min = kwargs_quasars.get('z_min', 0.1)
    z_max = kwargs_quasars.get('z_max', 5)
    m_min = kwargs_quasars.get('m_min', 17)
    m_max = kwargs_quasars.get('m_max', 23)

    redshifts = np.random.uniform(z_min, z_max, num_quasars)
    magnitude_r = np.random.uniform(m_min, m_max, num_quasars)
    magnitude_g = magnitude_r - 0.5
    magnitude_i = magnitude_r + 0.2

    point_source_catalog = Table(
        [redshifts, magnitude_r, magnitude_g, magnitude_i],
        names=("z", "mag_r", "mag_g", "mag_i"),
    )
    return point_source_catalog
