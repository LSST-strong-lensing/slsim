import numpy as np
from astropy.table import Table


def quasar_catalog_simple(
    num_quasars=50000,
    z_min=0.1,
    z_max=5,
    m_min=17,
    m_max=23,
    amp_min=0.9,
    amp_max=1.3,
    freq_min=0.5,
    freq_max=1.5,
):
    """Creates an simple catalog of quasars. It generate random redshift and magnitude
    values in r, g, and i band. Also, generates amplitude and frequency for each source.
    The function only works for r, g, and i band magnitudes.

    :param number: number of sources we want
    :param z_min: minimum redshift for sources
    :param z_max: maximum redshift for sources
    :param m_min: minimum magnitude for sources in r band
    :param m_max: maximum magnitude for sources in r band
    :param amp_min: minimum amplitude for sources
    :param amp_max: maximum amplitude for sources
    :param freq_min: minimum frequency for sources
    :param freq_max: maximum frequency for sources
    :return: an astropy table of quasar catalog
    """
    redshifts = np.random.uniform(z_min, z_max, num_quasars)
    magnitude_r = np.random.uniform(m_min, m_max, num_quasars)
    magnitude_g = magnitude_r - 0.5
    magnitude_i = magnitude_r + 0.2
    amplitude = np.random.uniform(amp_min, amp_max, num_quasars)
    frequency = np.random.uniform(freq_min, freq_max, num_quasars)

    point_source_catalog = Table(
        [redshifts, magnitude_r, magnitude_g, magnitude_i, amplitude, frequency],
        names=("z", "ps_mag_r", "ps_mag_g", "ps_mag_i", "amp", "freq"),
    )
    return point_source_catalog
