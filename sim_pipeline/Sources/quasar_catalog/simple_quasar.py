import numpy as np
from astropy.table import Table

def quasar_catalog(n, z_min, z_max, logL_min, logL_max):
    num_quasars = n
    redshifts = np.random.uniform(z_min, z_max, num_quasars)
    luminosities = np.random.uniform(logL_min, logL_max, num_quasars)

    # Calculate apparent magnitudes in r, g, and i bands
    # need appropriate distance and k-corrections here
    magnitude_r = -2.5 * np.log10(luminosities / (4 * np.pi * (10)**2))
    magnitude_g = magnitude_r - 0.5  
    magnitude_i = magnitude_r + 0.2

    point_source_catalog = Table([redshifts, magnitude_r, magnitude_g, magnitude_i],names=('z', 'mag_r', 'mag_g', 'mag_i'))
    return point_source_catalog