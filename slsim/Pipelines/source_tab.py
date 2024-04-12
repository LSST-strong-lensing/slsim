import numpy as np
import sys
from colossus.cosmology import cosmology
from scipy.stats import poisson

#
# make source table
#
# Define a function named make_srctab which takes in five arguments:
# mmax (maximum magnitude of source objects), fov (field of view),
# flag_type_min (minimum type of source object), flag_type_max (maximum type of source object),
# and cosmo (cosmology model)



def calc_vol(z, cosmo):
    dis = cosmo.angularDiameterDistance(z) / (cosmo.H0 / 100.0)
    drdz = (2997.92458 / ((1.0 + z) * cosmo.Ez(z))) / (cosmo.H0 / 100.0)

    # 3282.806350011744 is 1rad^2 = 3282.806350011744[deg^2]
    # multiply fov[deg^2] to obtain the expected number to be observed after by KA
    return (dis * dis / 3282.806350011744) * drdz * (1.0 + z) * (1.0 + z) * (1.0 + z)
