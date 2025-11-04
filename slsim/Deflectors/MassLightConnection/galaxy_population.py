from collections import OrderedDict

import numpy as np
import scipy.stats as st
from colossus.halo import mass_so


class GalaxySizeModel:
    """Characteristics of galaxy effective radius models.

        This object contains certain characteristics of a galaxy effective radius model.
    The :data:`models` dictionary contains one item of this class for each available model.
    """

    def __init__(self):
        """Initialize a new instance of the GalaxySizeModel class.

        Attributes:
        - 'func' : callable or None. The primary function that defines the model.
        - 'func_scat' : callable or None. An optional function to add scatter to the model.
        - 'mhalo_dependence' : bool. A flag indicating if there is a dependence on halo mass.
        - 'mstar_dependence' : bool. A flag indicating if there is a dependence on stellar mass.
        - 'z_dependence' : bool.  A flag indicating if there is a redshift dependence.
        - 'scatter' : bool. A flag indicating if scatter should be included in the model.
        - 'sc_sigtb_dependence' : bool. A flag indicating if there is a dependence on sigma_tb for scatter.
        - 'sc_mstar_dependence' : bool. A flag indicating if there is a stellar mass dependence for scatter.
        - 'sc_mhalo_dependence' : bool. A flag indicating if there is a halo mass dependence for scatter.
        - 'sc_z_dependence' : bool. A flag indicating if there is a redshift dependence for scatter.
        """
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


###################################################################################################


models = OrderedDict()
"""Dictionary containing a list of models.

An ordered dictionary containing one :class:`GalaxySizeModel` entry for each model.
"""

models["oguri20"] = GalaxySizeModel()
models["oguri20"].mhalo_dependence = True
models["oguri20"].z_dependence = True
models["oguri20"].sc_sigtb_dependence = True

models["vdW23"] = GalaxySizeModel()
models["vdW23"].mstar_dependence = True
models["vdW23"].z_dependence = True
models["vdW23"].sc_mstar_dependence = True

models["karmakar23"] = GalaxySizeModel()
models["karmakar23"].mhalo_dependence = True
models["karmakar23"].z_dependence = True
models["karmakar23"].sc_mhalo_dependence = True
models["karmakar23"].sc_z_dependence = True


def galaxy_size(
    mh, mstar, z, cosmo_col, q_out="tb", model="oguri20", scatter=False, sig_tb=0.1
):
    """Calculate the size of a galaxy based on halo mass, stellar mass,
    redshift, and cosmology, optionally including scatter.

    :param mh: The halo mass or masses at which to compute the galaxy size in units of M_sol/h
    :type  mh: float, np.ndarray, list
    :param mstar: The stellar mass or masses in units of M_sol/h
    :type  mstar: float, np.ndarray, list
    :param z: The redshift at which to compute the galaxy size.
    :type  z: float
    :param cosmo_col: An instance of a cosmology class to calculate angular diameter distances.
    :type  cosmo_col: Class (e.g., colossus.cosmology)
    :param q_out: The desired output quantity ('tb' for theta_b in arcsec or 'rb' for r_b in kpc/h).
    :type  q_out: str
    :param model: The name of the model used to calculate galaxy size ('oguri20' is default).
    :type  model: str
    :param scatter: Whether to include scatter in the galaxy size calculation.
    :type  scatter: bool
    :param sig_tb: The standard deviation of lognormal scatter if scatter is included for oguri20 model
    :type  sig_tb: float

    :return: Galaxy effective radius calculated according to the specified model and parameters.
             Units are either in arcseconds (if q_out='tb') or kpc/h (if q_out='rb'), np.ndarray or float

    Notes
    -----------------------------------------------------------------------------------------------
    This function supports multiple galaxy size models which may depend on different combinations
    of halo mass, stellar mass, and redshift. It can also apply lognormal scatter to the sizes.
    """

    # Check that the model exists
    if model not in models.keys():
        raise Exception("Unknown model, %s." % (model))
    if not isinstance(mh, (float, np.ndarray, list)):
        raise ValueError("type(mh) should be float, ndarray or list.")

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

    if q_out == "tb":
        convert_t = 1.0 / cosmo_col.angularDiameterDistance(z) * 206264.8
        gal_size = rb * convert_t
    elif q_out == "rb":
        gal_size = rb
    else:
        raise Exception("Unknown output, %s." % (q_out))

    if scatter:
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
            gal_size = gal_size * scat[0]
        else:
            args += (len(mh),)
            scat = model_props.func_scat(*args)
            gal_size = gal_size * scat

    return gal_size


###################################################################################################


def modelOguri20(mh, z):
    """The galaxy-size model of Oguri&Takahashi et al 2020.

    :param mh: The mass of the halo in units of M_sol/h
    :type  mh: ndarray
    :param z: The redshift
    :type  z: float
    :return: rb, float, The galaxy size, has the dimensions as 1.0e-3 *
        mass_so.M_to_R(mh, z, 'vir') i.e. [Mpc].
    """
    re = 0.006 * 1.0e-3 * mass_so.M_to_R(mh, 0.0, "vir") / (1.0 + z)
    rb = 0.551 * re

    return rb


def modelscLognormal(sig_tb, n):
    """Generate samples from a lognormal distribution with specified standard
    deviation and number of samples.

    :param sig_tb: The standard deviation of the lognormal distribution.
    :type  sig_tb: float
    :param n: The number of samples to generate.
    :type  n: int
    :return: Samples from a lognormal distribution. ndndarray
    """
    return np.random.lognormal(0.0, sig_tb, n)


def modelVanderwel23(mstar, z):
    """
    The galaxy-size model of van der Wel et al 2023.
    arXiv: 2307.03264
    re: effective (half-mass) radius

    re is calculated as follows

    .. math::

        \\log_{10}(r_e/(h^{-1}\\mathrm{kpc}))= \\Gamma + \alpha \\log_{10}(M_\\mathrm{cen}/M_\\odot) \\
            + (\beta-\alpha)\\log_{10}(1+10^{\\log_{10}(M_\\mathrm{cen}/M_\\odot)-\\delta} - \\omega \\log_{10}((1+z)/(1+z_\\mathrm{data}))

    where z_data= 0.75

    :param mstar: satellite or central galaxy mass [Msun/h]; can be a number or a numpy array.
                NOTICE: this mstar is based on Chabrier IMF. Then we use mstar_cor = mstar_chab/frac_SM_IMF as an input param
    :type  mstar: ndarray

    :param z: redshift; can be a number or a numpy array.
    :type  z: float

    :return: rb. ndarray or a number. The galaxy size, has the dimensions as 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') i.e. [Mpc/h].

    Notes
    -----------------------------------------------------------------------------------------------
    The following list of c_vdW50 shows [\\Gamma, \alpha, \beta, \\delta] appeared in the above equation. 
    These values are determined by applying the curve_fit function in scipy.optimize to the stellar half-mass radii data
    of the 50% percentile in Table 5 of van der Wel et al. 2023 (arXiv: 2307.03264), 
    which is for quiescent galaxies at 0.5 < z < 1.0.
    """
    c_vdW50 = [
        0.58302746,
        -0.06713236,
        1.1363604,
        10.81504371,
    ]  # [\Gamma, \alpha, \beta, \delta]
    re_wo_zdepend = (
        10
        ** log10Re_log10Mstar_vdW(
            np.log10(mstar), c_vdW50[0], c_vdW50[1], c_vdW50[2], c_vdW50[3]
        )
        / 1e3
    )  # [Mpc/h]
    omega = -1.72  # np.where(np.log10(mstar) < c_vdW50[3], -0.412, -1.72)
    zbin = (0.5 + 1.0) / 2.0
    zdepend = np.where(
        z > 2.5,
        ((1.0 + 2.5) / (1.0 + zbin)) ** omega,
        ((1.0 + z) / (1.0 + zbin)) ** omega,
    )
    log10ms_switch = 10.5
    re_lowmass_wo_zdepend = (
        10
        ** log10Re_log10Mstar_vdW(
            log10ms_switch, c_vdW50[0], c_vdW50[1], c_vdW50[2], c_vdW50[3]
        )
        / 1e3
    )  # [Mpc/h]
    re_lowconst_wo_zdepend = np.where(
        np.log10(mstar) > log10ms_switch, re_wo_zdepend, re_lowmass_wo_zdepend
    )
    re = re_lowconst_wo_zdepend * zdepend
    rb = 0.551 * re
    return rb


def modelscVanderwel23(mstar, n):
    """Generate samples of effective radii for galaxies based on a lognormal
    distribution using parameters from van der Wel et al.(2023).

    \\sigma_\\log(r_e) is calculated as follows

    .. math::
        \\sigma_\\log(r_e) = \\frac{\\log(r_e^{84}(z_\\mathrm{data}))-\\log(r_e^{16}(z_\\mathrm{data}))}{2}
    where
        \\log_{10}(r_e/(h^{-1}\\mathrm{kpc}))= \\Gamma + \alpha \\log_{10}(M_\\mathrm{cen}/M_\\odot) \\
            + (\beta-\alpha)\\log_{10}(1+10^{\\log_{10}(M_\\mathrm{cen}/M_\\odot)-\\delta}

    where z_data= 0.75, r_e^{84} and r_e^{16} are the values of effective radii of the 16\\%th and 84\\%th percentiles

    :param mstar: Stellar mass of galaxies for which to generate the effective radius in units of M_sol/h
    :type  mstar: np.ndarray

    :param n: The number of samples
    :type  n: int

    :return: Scatters of galaxy effective radii from the lognormal distribution. ndndarray

    Notes
    -----------------------------------------------------------------------------------------------
    The following lists of c_vdW84 and c_vdW16 show [\\Gamma, \alpha, \beta, \\delta] appeared in the above equation.
    These values are determined by applying the curve_fit function in scipy.optimize to the stellar half-mass radii data
    of the 84% and 16% percentiles in Table 5 of van der Wel et al. 2023 (arXiv: 2307.03264), which are for quiescent galaxies at 0.5 < z < 1.0.
    mstar_cor is defined to prevent the scatter from becoming too small or negative at the high mass end considering the fitting function.
    The criteria mass of 10**11.4 mostly corresponds the maximum limit of data sets in van der Wel et al. 2023.
    """
    c_vdW84 = [
        0.64141456,
        -0.05489086,
        1.02386427,
        10.79889608,
    ]  # [\Gamma, \alpha, \beta, \delta]
    c_vdW16 = [
        0.77059797,
        -0.1087621,
        1.18547984,
        10.68959868,
    ]  # [\Gamma, \alpha, \beta, \delta]
    mstar_cor = np.where(
        mstar > 10**11.4, 10**11.4, mstar
    )  # to prevent the scatter from becoming too small or negative at the high mass end
    log10Re_vdW84_pre = log10Re_log10Mstar_vdW(
        np.log10(mstar_cor), c_vdW84[0], c_vdW84[1], c_vdW84[2], c_vdW84[3]
    )
    log10Re_vdW16_pre = log10Re_log10Mstar_vdW(
        np.log10(mstar_cor), c_vdW16[0], c_vdW16[1], c_vdW16[2], c_vdW16[3]
    )

    log10Re_vdW84_lowmass = log10Re_log10Mstar_vdW(
        c_vdW84[3], c_vdW84[0], c_vdW84[1], c_vdW84[2], c_vdW84[3]
    )
    log10Re_vdW16_lowmass = log10Re_log10Mstar_vdW(
        c_vdW16[3], c_vdW16[0], c_vdW16[1], c_vdW16[2], c_vdW16[3]
    )

    log10Re_vdW16 = np.where(
        np.log10(mstar_cor) > c_vdW16[3],
        log10Re_vdW16_pre,
        log10Re_vdW16_lowmass,
    )
    log10Re_vdW84 = np.where(
        np.log10(mstar_cor) > c_vdW84[3],
        log10Re_vdW84_pre,
        log10Re_vdW84_lowmass,
    )
    ave_1sigma = (log10Re_vdW84 - log10Re_vdW16) / 2.0 * np.log(10)
    return np.random.lognormal(0.0, ave_1sigma, n)


def modelKarmakar23(mh, z):
    """
    The galaxy-size model of T. Karmakar et al 2023.
    arXiv: 2301.10409
    We here adopt the following double-power-law fitting function as one of simple fitting functions,

    .. math::
        r_e/R_\\mathrm{vir} = a*\\frac{M_h}{10^{12} \\mathrm{M_\\odot/h}}**b*(0.5*(1
        +\\frac{M_h}{10^{12} \\mathrm{M_\\odot/h}}^6))^{0.001-b/6}

    where :math:`r_e` and :math:`R_\\mathrm{vir}` are galaxy effective radius and halo virial radius.

    :param mh: (sub)halo mass [Msun/h]; can be a number or a numpy array.
    :type  mh: ndarray
    :param z: redshift; can be a number or a numpy array.
    :type  z: ndarray
    :return: rb. ndarray or a number. The galaxy size, has the dimensions as 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') i.e. [Mpc].

    Notes
    -----------------------------------------------------------------------------------------------
    The following values of a_z and b_z are determined by applying the curve_fit function in scipy.optimize
    to the ratio between :math:`r_e` and :math:`R_\\mathrm{vir}` for the data at four redshifts (z=0, 1, 2, 3) in T. Karmakar et al. (2023)
    with the above fitting function and then applying numpy.polyfit at a linear level as functions of z.
    """
    a_z = -0.00135984 * z + 0.01667855
    b_z = -0.07948921 * z - 0.23212207
    d_z = 1e12
    c_z = 0.001

    reRh = a_z * (mh / d_z) ** b_z * (0.5 * (1 + (mh / d_z) ** 6)) ** ((c_z - b_z) / 6)
    re = reRh * 1.0e-3 * mass_so.M_to_R(mh, z, "vir")
    rb = 0.551 * re
    return rb


def modelscKarmakar23(mh, z, n):
    """Generate a log-normal distribution of the scaling relation parameter based on
    halo mass and redshift from  T. Karmakar et al 2023.
    arXiv: 2301.10409
    We here adopt the following double-power-law fitting function as one of simple fitting functions,

    .. math::
        r_e/R_\\mathrm{vir} = a*\\frac{M_h}{d}**b*(0.5*(1+\\frac{M_h}{d)^{c-b/6}

    where :math:`r_e` and :math:`R_\\mathrm{vir}` are galaxy effective radius and halo virial radius.

    :param mh: The halo mass or masses used in the scaling relation in units of M_sol/h
    :type  mh: float or np.ndarray
    :param z: The redshift value used in the scaling relation.
    :type  z: float
    :param n: The number of samples to draw from the distribution.
    :type  n: int

    :return: An array of values drawn from a log-normal distribution defined by the scaling relation. ndndarray

    Notes
    -----------------------------------------------------------------------------------------------
    The following values of a_z, b_z, c_z, and d_z are determined by applying the curve_fit function in scipy.optimize
    to the ratio between :math:`r_e` and :math:`R_\\mathrm{vir}` for the data at four redshifts (z=0, 1, 2, 3) in T. Karmakar et al. (2023)
    with the above fitting function and then applying numpy.polyfit at a linear level as functions of z.
    d is in units of M_sol/h
    """
    a_z = 0.03461388 * z + 0.16207918
    b_z = -0.00304315 * z + 0.0265449
    c_z = -0.06415788 * z - 0.20405057
    d_z = 2.41180793e10 * z + 9.42953770e11
    sig = a_z * (mh / d_z) ** b_z * (0.5 * (1 + (mh / d_z) ** 6)) ** ((c_z - b_z) / 6)
    return np.random.lognormal(0.0, sig, n)


###################################################################################################

###################################################################################################
# Pointers to model functions
###################################################################################################


models["oguri20"].func = modelOguri20
models["oguri20"].func_scat = modelscLognormal
models["vdW23"].func = modelVanderwel23
models["vdW23"].func_scat = modelscVanderwel23
models["karmakar23"].func = modelKarmakar23
models["karmakar23"].func_scat = modelscKarmakar23


def log10Re_log10Mstar_vdW(log10M, a, b, c, d):
    """Function to calculate the logarithm of the effective radius as a
    function of the logarithm of the stellar mass used in vdw23 model.

    :param log10M: The logarithm (base 10) of the stellar mass.
    :type  log10M: float
    :param a: Coefficient for the constant term in the relation.
    :type  a: float
    :param b: Coefficient for the linear term in the relation.
    :type  b: float
    :param c: Coefficient defining the curvature of the relation.
    :type  c: float
    :param d: Characteristic mass where the curvature changes.
    :type  d: float
    :return: log10Re. float. The logarithm (base 10) of the effective
        radius computed using the given parameters.
    """
    return a + b * log10M + (c - b) * np.log10(1 + 10 ** (log10M - d)) ** (c - b)


class p_smhm:
    """Characteristics of stellar mass halo mass relation models.

    based on the fitting function for the stellar mass - halo mass relation in Behroozi+ 2019
    """

    def __init__(self, data):
        """
        Initialize the p_smhm class with parameters for stellar mass - halo mass relation.

        :param data: A list or array containing the values of the parameters that define the stellar mass-halo mass relation.
        :type  data: list or numpy.array

        Notes
        -----------------------------------------------------------------------------------------------
        The following parameters are fitting parameters of the stellar-mass halo-mass relation in Behroozi et al. 2019 (arXiv: 1806.07893),
        which include:

        - 'ep0' : epsilon_0, constant term in the epsilon parameter evolution equation.
        - 'epa' : epsilon_a, scale factor dependent term in the epsilon parameter evolution equation.
        - 'eplna' : epsilon_{lna}, natural log of the scale factor term in the epsilon parameter evolution equation.
        - 'epz': epsilon_z, redshift dependent term in the epsilon parameter evolution equation.

        - 'M0' : M_0, constant term in the logarithm base 10 of M1 over solar mass equation.
        - 'Ma' : M_a, scale factor dependent term in the logarithm base 10 of M1 over solar mass equation.
        - 'Mlna' : M_{lna}, natural log of the scale factor term in the logarithm base 10 of M1 over solar mass equation.
        - 'Mz' : M_z, redshift dependent term in the logarithm base 10 of M1 over solar mass equation.

        - 'alpha0' : alpha_0, constant term in the alpha parameter evolution equation.
        - 'alphaa' : alpha_a, scale factor dependent term in the alpha parameter evolution equation.
        - 'alphalna' : alpha_{lna}, natural log of the scale factor term in the alpha parameter evolution equation.
        - 'alphaz' : alpha_z, redshift dependent term in the alpha parameter evolution equation.

        - 'beta0' : beta_0, constant term in the beta parameter evolution equation.
        - 'betaa' : beta_a, scale factor dependent term in the beta parameter evolution equation.
        - 'betaz' : beta_z, redshift dependent term in the beta parameter evolution equation.

        - 'delta0' : delta_0, constant term representing the delta parameter, which is assumed to be constant.

        - 'gamma0' : gamma_0, constant term in the logarithm base 10 of the gamma parameter evolution equation.
        - 'gammaa' : gamma_a, scale factor dependent term in the logarithm base 10 of the gamma parameter evolution equation.
        - 'ammaz' : gamma_z, redshift dependent term in the logarithm base 10 of the gamma parameter evolution equation.

        The equations represented by these parameters are as follows:

        .. math::
            \\log _{10}\\left(\frac{M_*}{M_1}\right)= & \\epsilon-\\log _{10}\\left(10^{-\alpha x}+10^{-\beta x}\right)
            +\\gamma \\exp \\left[-0.5\\left(\frac{x}{\\delta}\right)^2\right],\\
            \\(\\log _{10}\\left(\frac{M_1}{M_{\\odot}}\right) = M_0 + M_a(a - 1) - M_{\\ln a} \\ln(a) + M_z z,\\)\\
            \\(\\epsilon = \\epsilon_0 + \\epsilon_a(a - 1) - \\epsilon_{\\ln a} \\ln(a) + \\epsilon_z z,\\)\\
            \\(\alpha = \alpha_0 + \alpha_a(a - 1) - \alpha_{\\ln a} \\ln(a) + \alpha_z z,\\)\\
            \\(\beta = beta_0 + beta_a(a - 1) + beta_z z,\\)\\
            \\(\\delta = delta_0,\\)\\
            \\(\\log _{10}(\\gamma) = gamma_0 + gamma_a(a - 1) + gamma_z z.\\)\\

        where M_* is the stellar mass with the Chabrier initial mass function.
        """
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
    """Set the galaxy parameters based on the halo position angle.

    :param pol_halo: Array or list of halo position angle values.
    :type  pol_halo: list or numpy.array
    :return: elip_gal and polar_gal. An array of galaxy ellipticities
        generated for each halo and an array of galaxy position anglen
        angles derived from the halo position angle
    """
    n = len(pol_halo)
    elip_gal = gene_e(n)
    polar_gal = gene_ang_gal(pol_halo)
    return elip_gal, polar_gal


def gals_init(TYPE_SMHM="true"):
    """
    The fitting parameters for calculating stellar-mass-halo-mass function of P. Behroozi et al. 2019.
    arXiv: 1806.07893

    :param TYPE_SMHM: "true" and "obs" are for quiescent galaxies but "true_all" is for all galaxies
                    in ["true", "obs", "true_all"]
    :type  TYPE_SMHM:  str
    :return: paramc, params. list. The fitting parameters for central galaxies and satellite galaxies
        For the meaning of each parameters, please see the Notes in p_smhm.__init__

    Notes
    -----------------------------------------------------------------------------------------------
    The following values are summarized in Table J1 of P. Behroozi et al. 2019 (arXiv: 1806.07893)
    The values with TYPE_SMHM == "true" show the true values for quenched galaxies from Markov Chain Monte Carlo calculation in P. Behroozi et al. 2019
    The values with TYPE_SMHM == "obs" show the values from observations for quenched galaxies
    The values with TYPE_SMHM == "true_all" show the true values for all galaxies including star-forming and quenched galaxies
    """

    if TYPE_SMHM == "true":
        p_smhm_cen = [
            -1.462,
            -0.732,
            -1.273,
            0.302,
            12.072,
            3.581,
            3.665,
            -0.634,
            1.928,
            -3.472,
            -3.119,
            0.507,
            0.488,
            -0.419,
            -0.256,
            0.406,
            -0.980,
            -1.443,
            -0.335,
        ]  # true, quench
        p_smhm_sat = [
            -1.432,
            -1.231,
            -0.999,
            0.100,
            11.889,
            3.236,
            3.378,
            -0.577,
            1.959,
            -4.033,
            -3.175,
            0.390,
            0.464,
            0.130,
            -0.153,
            0.319,
            -0.812,
            0.522,
            0.064,
        ]  # true, quench
    elif TYPE_SMHM == "obs":
        p_smhm_cen = [
            -1.480,
            -0.831,
            -1.351,
            0.321,
            12.069,
            2.646,
            2.710,
            -0.431,
            1.899,
            -2.901,
            -2.413,
            0.332,
            0.502,
            -0.315,
            -0.218,
            0.397,
            -0.867,
            -1.146,
            -0.294,
        ]  # observation, quench
        p_smhm_sat = [
            -1.449,
            -1.256,
            -1.031,
            0.108,
            11.896,
            3.284,
            3.413,
            -0.580,
            1.949,
            -4.096,
            -3.226,
            0.401,
            0.477,
            0.046,
            -0.214,
            0.357,
            -0.755,
            0.461,
            0.025,
        ]  # observation, all(Q/SF)
    elif TYPE_SMHM == "true_all":
        p_smhm_cen = [
            -1.431,
            1.757,
            1.350,
            -0.218,
            12.074,
            4.600,
            4.423,
            -0.732,
            1.974,
            -2.468,
            -1.816,
            0.182,
            0.470,
            -0.875,
            -0.487,
            0.382,
            -1.160,
            -3.634,
            -1.219,
        ]  # true, all(Q/SF)
        p_smhm_sat = [
            -1.432,
            -1.231,
            -0.999,
            0.100,
            11.889,
            3.236,
            3.378,
            -0.577,
            1.959,
            -4.033,
            -3.175,
            0.390,
            0.464,
            0.130,
            -0.153,
            0.319,
            -0.812,
            0.522,
            0.064,
        ]  # true, all(Q/SF)

    paramc = p_smhm(p_smhm_cen)
    params = p_smhm(p_smhm_sat)
    return paramc, params


def stellarmass_halomass(Mh, z, pa, frac_SM_IMF=1.715):
    """
    Calculate the stellar mass of a galaxy from its halo mass using an empirical relation.
    see P. Behroozi et al. 2019. for detail
    arXiv: 1806.07893

    :param Mh: Halo mass of the galaxy in units of M_sol/h
    :type  Mh: float
    :param z: The redshift at which the stellar mass is calculated.
    :type  z: float
    :param pa: Parameter set for the stellar-mass halo-mass relation.
    :type  pa: object
    :param frac_SM_IMF: Fraction of the stellar mass due to the initial mass function (IMF) against Chabrier IMF.
                        Default value is set to 1.715, coming from Salpeter IMF. (set to 1.0 for Chabrier IMF)
    :type  frac_SM_IMF: float
    :return: stellar_mass. float. The estimated stellar mass of the galaxy in units of M_sol/h
    """
    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = np.log(a)
    m_1 = pa.M0 + a1 * pa.Ma - lna * pa.Mlna + z * pa.Mz
    stellarm_0 = m_1 + pa.ep0 + a1 * pa.epa - lna * pa.eplna + z * pa.epz
    alpha = pa.alpha0 + a1 * pa.alphaa - lna * pa.alphalna + z * pa.alphaz
    beta = pa.beta0 + a1 * pa.betaa + z * pa.betaz
    delta = pa.delta0
    gamma = 10.0 ** (pa.gamma0 + a1 * pa.gammaa + z * pa.gammaz)
    x = np.log10(Mh) - m_1
    x_del = x / delta
    stellarm = (
        stellarm_0
        - np.log10(10.0 ** (-alpha * x) + 10.0 ** (-beta * x))
        + gamma * np.exp(-0.5 * (x_del**2))
    )
    return 10**stellarm * frac_SM_IMF


def gene_e(n):
    """
    Generate an ellipticity distribution for a sample of galaxies in M. Oguri et al. 2008
    arXiv: 0708.0825

    :param n: Number of galaxies in the sample for which to generate ellipticities.
    :type  n: int

    :return: elipticity of 1-q, where q is axis ratio. ndarray. An array of galaxy ellipticities drawn from a truncated normal distribution.
    """

    em = 0.3
    se = 0.16
    e = st.truncnorm.rvs((0.0 - em) / se, (0.9 - em) / se, loc=em, scale=se, size=n)
    return e


def gene_ang_gal(pol_h):
    """
    Position angle of the galaxies relative to the major axis of the  halos in T. Okumura et al. 2009
    arXiv: 0809.3790

    :param pol_h: position angles of halos
    :type  pol_h: ndarray
    :return: pol_gal. ndarray. position angle of galaxies
    """
    n = len(pol_h)
    sig = 35.4
    pol_gal = np.random.normal(loc=pol_h, scale=sig, size=n)
    return pol_gal


#
# for checks
#
if __name__ == "__main__":
    print(p_smhm.__init__.__doc__)
