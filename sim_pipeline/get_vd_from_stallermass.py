import numpy as np


def get_vd_from_stallermass(smass):
    """
    function for calculate the velocity dispersion from the staller mass using empirical relation

    :param smass: stellar mass in the unit of solar mass,

    return vdmass: the velocity dispersion ("km/s")
    2.32,0.24 is the parameters from [1] table 2
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and total mass correlations of massive early-type galaxies." The Astrophysical Journal 724.1 (2010): 511.
    """
    stellarmass = smass
    vdmass = (np.power(10,2.32) * np.power(stellarmass/1e11, 0.24))
    return vdmass