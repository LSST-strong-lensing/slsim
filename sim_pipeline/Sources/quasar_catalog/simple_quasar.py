import numpy as np
from astropy.table import Table

def quasar_catalog(n, z_min, z_max, m_min, m_max):
    num_quasars = n
    redshifts = np.random.uniform(z_min, z_max, num_quasars)
    magnitude_r = np.random.uniform(m_min, m_max, num_quasars)
    magnitude_g = magnitude_r - 0.5  
    magnitude_i = magnitude_r + 0.2
    n_sersic_single = 4
    repeats = num_quasars
    n_sersic = [n_sersic_single for _ in range(repeats)]

    point_source_catalog = Table([redshifts, magnitude_r, magnitude_g, magnitude_i, n_sersic],
                                 names=('z', 'mag_r', 'mag_g', 'mag_i', 'n_sersic'))
    return point_source_catalog