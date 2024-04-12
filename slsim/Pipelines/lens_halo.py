import numpy as np

from scipy import interpolate
from colossus.lss import mass_function
from colossus.halo import concentration
#
# number counts and properties of lens halo
#


def calc_vol(z, cosmo_col):
    dis = cosmo_col.angularDiameterDistance(z) / (cosmo_col.H0 / 100.0)
    drdz = (2997.92458 / ((1.0 + z) * cosmo_col.Ez(z))) / (cosmo_col.H0 / 100.0)

    # 3282.806350011744 is 1rad^2 = 3282.806350011744[deg^2]
    # multiply fov[deg^2] to obtain the expected number to be observed after by KA
    return (dis * dis / 3282.806350011744) * drdz * (1.0 + z) * (1.0 + z) * (1.0 + z)

def dNhalodzdlnM_lens(M, z, cosmo_col):#cosmo in lenstronomy
    dvoldzdO = calc_vol(z, cosmo_col)
    hhh = (cosmo_col.H0 / 100.0)**3
    mfunc_so = mass_function.massFunction(
        M, z, mdef='fof', model='sheth99', q_out='dndlnM')*hhh
    return dvoldzdO*mfunc_so  # [number/deg^2/dlnM[Modot/h]]


def concent_m_w_scatter(m, z, sig):
    con = concentration.concentration(m, 'vir', z, model='diemer15')
    sca = np.random.lognormal(0.0, sig, len(m))
    return con*sca
