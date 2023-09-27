import numpy as np
from astropy.table import Table


def quasar_catalog(n, z_min, z_max, m_min, m_max):
    num_quasars = n
    redshifts = np.random.uniform(z_min, z_max, num_quasars)
    magnitude_r = np.random.uniform(m_min, m_max, num_quasars)
    magnitude_g = magnitude_r - 0.5
    magnitude_i = magnitude_r + 0.2

    point_source_catalog = Table(
        [redshifts, magnitude_r, magnitude_g, magnitude_i],
        names=("z", "mag_r", "mag_g", "mag_i"),
    )
    return point_source_catalog
