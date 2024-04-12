import numpy as np
from colossus.halo import mass_so
from collections import OrderedDict
import solve_lenseq

class GalaxySizeModel():
    def __init__(self):
        self.func = None
        self.func_scat = None
        self.mhalo_dependence = False
        self.mstar_dependence = False
        self.z_dependence = False
        self.scatter = False
        self.sc_sigtb_dependence = False
        self.sc_mstar_dependence = False
        self.sc_mhalo_dependence = False
        self.sc_z_dependence = False
        return

###################################################################################################


models = OrderedDict()
"""
Dictionary containing a list of models.

An ordered dictionary containing one :class:`GalaxySizeModel` entry for each model.
"""

models['oguri20'] = GalaxySizeModel()
models['oguri20'].mhalo_dependence = True
models['oguri20'].z_dependence = True
models['oguri20'].sc_sigtb_dependence = True

models['vdW23'] = GalaxySizeModel()
models['vdW23'].mstar_dependence = True
models['vdW23'].z_dependence = True
models['vdW23'].sc_mstar_dependence = True

models['karmakar23'] = GalaxySizeModel()
models['karmakar23'].mhalo_dependence = True
models['karmakar23'].z_dependence = True
models['karmakar23'].sc_mhalo_dependence = True
models['karmakar23'].sc_z_dependence = True


def galaxy_size(mh, mstar, z, cosmo, q_out='tb', model='oguri20', scatter=False, sig_tb=0.1):

    # Check that the model exists
    if not model in models.keys():
        raise Exception('Unknown model, %s.' % (model))

    model_props = models[model]

    # Create the argument list depending on the model and evaluate it.
    args = ()
    if model_props.mhalo_dependence:
        args += (mh,)
    if model_props.mstar_dependence:
        args += (mstar,)
    if model_props.z_dependence:
        args += (z,)

    rb = model_props.func(*args)

    if q_out == 'tb':
        convert_t = 1./cosmo.angularDiameterDistance(z)*206264.8
        gal_size = rb*convert_t
    elif q_out == 'rb':
        gal_size = rb
    else:
        raise Exception('Unknown model, %s.' % (q_out))

    if scatter == True:
        args = ()
        if model_props.sc_sigtb_dependence:
            args += (sig_tb,)
        if model_props.sc_mstar_dependence:
            args += (mstar,)
        if model_props.sc_mhalo_dependence:
            args += (mh,)
        if model_props.sc_z_dependence:
            args += (z,)
        if isinstance(mh, float):
            args += (1,)
            scat = model_props.func_scat(*args)
#             gal_size = gal_size*np.random.lognormal(0.0, sig_tb)
            gal_size = gal_size*scat[0]
        elif isinstance(mh, (np.ndarray, list)):
            args += (len(mh),)
            scat = model_props.func_scat(*args)
            gal_size = gal_size*scat
        else:
            raise ValueError("type(mh) should be float, ndarray or list.")

    return gal_size

###################################################################################################


def modelOguri20(mh, z):
    """
    The galaxy-size model of Oguri&Takahashi et al 2020.

    Parameters
    -----------------------------------------------------------------------------------------------
    mh: ndarray
        (sub)halo mass [Msun/h]; can be a number or a numpy array.
    z:  ndarray
        redshift; can be a number or a numpy array.

    Returns
    -----------------------------------------------------------------------------------------------
    rb: ndarray or a number
        The galaxy size, has the dimensions as 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') i.e. [Mpc].
    """
    re = 0.006 * 1.0e-3 * mass_so.M_to_R(mh, 0.0, 'vir') / (1.0 + z)
    rb = 0.551*re

    return rb


def modelscLognormal(sig_tb, n):
    return np.random.lognormal(0.0, sig_tb, n)



def modelVanderwel23(mstar,z):
    """
    The galaxy-size model of A. Van Der Wel et al 2023.
    arXiv: 2307.03264
    re: effective (half-mass) radius

    Parameters
    -----------------------------------------------------------------------------------------------
    mstar: ndarray
        satellite or central galaxy mass [Msun/h]; can be a number or a numpy array.
        NOTICE: this mstar is based on Chabrier IMF. Then we use mstar_cor = mstar_chab/frac_SM_IMF as an input param
    z:  ndarray
        redshift; can be a number or a numpy array.

    Returns
    -----------------------------------------------------------------------------------------------
    rb: ndarray or a number
        The galaxy size, has the dimensions as 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') i.e. [Mpc].
    """
    c_vdW50 = [ 0.58302746, -0.06713236,  1.1363604,  10.81504371] #for [Msun/h] -> log10[kpc/h]
    re_wo_zdepend = 10**log10Re_log10Mh_vdW(np.log10(mstar),c_vdW50[0],c_vdW50[1], c_vdW50[2], c_vdW50[3])/1e3 #[Mpc/h]
    alpha = np.where(np.log10(mstar)<c_vdW50[3], -0.412, -1.72)
    zbin = (0.5+1.0)/2.0
    zdepend =  np.where(z>2.5,((1.+2.5)/(1.+zbin))**alpha,((1.+z)/(1.+zbin))**alpha)
    re = re_wo_zdepend*zdepend
    rb = 0.551*re
    return rb

def modelscVanderwel23(mstar,n):
    c_vdW84 = [ 0.64141456, -0.05489086, 1.02386427, 10.79889608]
    c_vdW16 = [ 0.77059797, -0.1087621,  1.18547984, 10.68959868]
    mstar_cor = np.where(mstar>10**11.43033199, 10**11.43033199, mstar) #to prevent the scatter from becoming too small or negative at the high mass end
    log10Re_vdW84 = log10Re_log10Mh_vdW(np.log10(mstar_cor),c_vdW84[0],c_vdW84[1], c_vdW84[2], c_vdW84[3])
    log10Re_vdW16 = log10Re_log10Mh_vdW(np.log10(mstar_cor),c_vdW16[0],c_vdW16[1], c_vdW16[2], c_vdW16[3])
    ave_1sigma = (log10Re_vdW84-log10Re_vdW16)/2.0*np.log(10)
    return np.random.lognormal(0.0, ave_1sigma, n)



def modelKarmakar23(mh, z):
    """
    The galaxy-size model of T. Karmakar et al 2023.
    arXiv: 2301.10409

    Parameters
    -----------------------------------------------------------------------------------------------
    mh: ndarray
        (sub)halo mass [Msun/h]; can be a number or a numpy array.
    z:  ndarray
        redshift; can be a number or a numpy array.

    Returns
    -----------------------------------------------------------------------------------------------
    rb: ndarray or a number
        The galaxy size, has the dimensions as 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') i.e. [Mpc].
    """
    a_z = -0.00135984*z + 0.01667855
    b_z = -0.07948921*z - 0.23212207
    d_z = 1e12
    c_z = 0.001

    reRh = a_z*(mh/d_z)**b_z*(0.5*(1+(mh/d_z)**6))**((c_z-b_z)/6)
    re = reRh*1.0e-3*mass_so.M_to_R(mh, z, "vir")
    rb = 0.551*re
    return rb


def modelscKarmakar23(mh, z, n):
    a_z = 0.03461388*z + 0.16207918
    b_z = -0.00304315*z + 0.0265449
    c_z = -0.06415788*z - 0.20405057
    d_z = 2.41180793e+10*z + 9.42953770e+11
    sig = a_z*(mh/d_z)**b_z*(0.5*(1+(mh/d_z)**6))**((c_z-b_z)/6)
    return np.random.lognormal(0.0, sig, n)

###################################################################################################

###################################################################################################
# Pointers to model functions
###################################################################################################


models['oguri20'].func = modelOguri20
models['oguri20'].func_scat = modelscLognormal
models['vdW23'].func = modelVanderwel23
models['vdW23'].func_scat = modelscVanderwel23
models['karmakar23'].func = modelKarmakar23
models['karmakar23'].func_scat = modelscKarmakar23


def log10Re_log10Mh_vdW(log10M,a,b,c,d):
    return a+b*log10M + (c-b)*np.log10(1+10**(log10M-d))**(c-b)

# ### Stellar mass - Halo mass relation from Behroozi+ 2019

class p_smhm:
    def __init__(self, data):
        self.ep0 = data[0]
        self.epa = data[1]
        self.eplna = data[2]
        self.epz = data[3]
        self.M0 = data[4]
        self.Ma = data[5]
        self.Mlna = data[6]
        self.Mz = data[7]
        self.alpha0 = data[8]
        self.alphaa = data[9]
        self.alphalna = data[10]
        self.alphaz = data[11]
        self.beta0 = data[12]
        self.betaa = data[13]
        self.betaz = data[14]
        self.delta0 = data[15]
        self.gamma0 = data[16]
        self.gammaa = data[17]
        self.gammaz = data[18]


def set_gals_param(pol_halo):
    n = len(pol_halo)
    elip_gal = solve_lenseq.gene_e(n)
    polar_gal = solve_lenseq.gene_ang_gal(pol_halo)
    return elip_gal, polar_gal


def set_param_smhm(data):
    p_smhm = {'ep0': data[0], 'epa': data[1], 'eplna': data[2], 'epz': data[3]}
    return p_smhm


def gals_init(TYPE_SMHM="true"):

    if (TYPE_SMHM == "true"):
        p_smhm_cen = [-1.462, -0.732, -1.273, 0.302, 12.072, 3.581, 3.665, -0.634,
                      1.928, -3.472, -3.119, 0.507, 0.488, -0.419, -0.256, 0.406, -0.980, -1.443, -0.335]  # true, quench
        p_smhm_sat = [-1.432, -1.231, -0.999, 0.100, 11.889,  3.236, 3.378, -0.577,
                      1.959, -4.033, -3.175, 0.390, 0.464, 0.130, -0.153, 0.319, -0.812, 0.522, 0.064]  # true, all(Q/SF)
    elif (TYPE_SMHM == "obs"):
        p_smhm_cen = [-1.480, -0.831, -1.351, 0.321, 12.069, 2.646, 2.710, -0.431,
                      1.899, -2.901, -2.413, 0.332, 0.502, -0.315, -0.218, 0.397, -0.867, -1.146, -0.294]  # observation, quench
        p_smhm_sat = [-1.449, -1.256, -1.031, 0.108, 11.896, 3.284, 3.413, -0.580,
                      1.949, -4.096, -3.226, 0.401, 0.477, 0.046, -0.214, 0.357, -0.755, 0.461, 0.025]  # observation, all(Q/SF)
    elif (TYPE_SMHM == "true_all"):
        p_smhm_cen = [-1.431, 1.757, 1.350, -0.218, 12.074, 4.600, 4.423, -0.732,
                      1.974, -2.468, -1.816, 0.182, 0.470, -0.875, -0.487, 0.382, -1.160, -3.634, -1.219]  # true, all(Q/SF)
        p_smhm_sat = [-1.432, -1.231, -0.999, 0.100, 11.889,  3.236, 3.378, -0.577,
                      1.959, -4.033, -3.175, 0.390, 0.464, 0.130, -0.153, 0.319, -0.812, 0.522, 0.064]  # true, all(Q/SF)

    paramc = p_smhm(p_smhm_cen)
    params = p_smhm(p_smhm_sat)
    return paramc, params


def stellarmass_halomass(Mh, z, pa, frac_SM_IMF=1.715):
    a = 1./(1.+z)
    a1 = a-1.
    lna = np.log(a)
    m_1 = pa.M0 + a1*pa.Ma - lna*pa.Mlna + z*pa.Mz
    stellarm_0 = m_1 + pa.ep0 + a1*pa.epa - lna*pa.eplna + z*pa.epz
    alpha = pa.alpha0 + a1*pa.alphaa - lna*pa.alphalna + z*pa.alphaz
    beta = pa.beta0 + a1*pa.betaa + z*pa.betaz
    delta = pa.delta0
    gamma = 10.**(pa.gamma0 + a1*pa.gammaa + z*pa.gammaz)
    x = np.log10(Mh)-m_1
    x_del = x/delta
    stellarm = stellarm_0 - \
        np.log10(10.**(-alpha*x) + 10.**(-beta*x)) + \
        gamma*np.exp(-0.5*(x_del**2))
    return 10**stellarm*frac_SM_IMF


