import numpy as np
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from astropy.table import Column


def quasar_galaxies_simple(
    m_min=17,
    m_max=23,
    amp_min=0.9,
    amp_max=1.3,
    freq_min=0.5,
    freq_max=1.5,
    sky_area=Quantity(value=0.1, unit="deg2"),
):
    """Creates an simple catalog of quasars and host galaxies. It generate random
    magnitude values in r, g, and i band for quasars. Also, generates amplitude and
    frequency for each source. Then, adds all these quasar properties to the galaxy
    catalog produced by skypy.

    :param z_max: maximum redshift for sources
    :param m_min: minimum magnitude for sources in r band
    :param m_max: maximum magnitude for sources in r band
    :param amp_min: minimum amplitude for sources
    :param amp_max: maximum amplitude for sources
    :param freq_min: minimum frequency for sources
    :param freq_max: maximum frequency for sources
    :return: an astropy table of quasars and host galaxies catalog
    """
    pipeline = SkyPyPipeline(
        skypy_config=None,
        sky_area=sky_area,
        filters=None,
        cosmo=None,
    )
    catalog = pipeline.red_galaxies
    magnitude_r = np.random.uniform(m_min, m_max, len(catalog))
    magnitude_g = magnitude_r - 0.5
    magnitude_i = magnitude_r + 0.2
    amplitude = np.random.uniform(amp_min, amp_max, len(catalog))
    frequency = np.random.uniform(freq_min, freq_max, len(catalog))

    mag_r_col = Column(name="ps_mag_r", data=magnitude_r)
    mag_g_col = Column(name="ps_mag_g", data=magnitude_g)
    mag_i_col = Column(name="ps_mag_i", data=magnitude_i)
    freq_col = Column(name="freq", data=frequency)
    amp_col = Column(name="amp", data=amplitude)

    catalog.add_columns([mag_r_col, mag_g_col, mag_i_col, amp_col, freq_col])
    return catalog
