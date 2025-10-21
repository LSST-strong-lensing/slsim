import numpy as np
import scipy.stats as st
from colossus.halo import concentration
from colossus.lss import mass_function

# Changed "vir" to 200c in concent_m_w_scatter function, mdef="vir" is not supported in diemer19 model.


def gene_e_ang_halo(Mh):
    """Ellipticity.

     .. math::
        ellipticity = \\equic \\1 - q

    where q is axis ratio
    and Position angle of the halo.

    :param Mh: halo mass
    :type  Mh: ndarray
    :return:  e, p, ellipticity of 1-q, where q is axis ratio and position angle of halos.
    """
    n = len(Mh)
    e = gene_e_halo(Mh)
    p = gene_ang(n)
    return e, p


def gene_ang(n):
    """Position angle of the halo.

    :param n: length of halo mass nd.array
    :type  n: int
    :return: position angle of halos in units of degree
    """
    return (np.random.rand(n) - 0.5) * 360.0


def gene_e_halo(Mh):
    """Ellipticity.

     .. math::
        ellipticity = \\equic \\1 - q

    of halos in T. Okabe et al. 2020
    arxiv: 2005.11469

    :param Mh: halo mass
    :type Mh: ndarray
    :return: ellipticity of halos
    """
    log10Mh = np.log10(Mh)  # log10([Modot/h])
    elip_fit = (
        0.09427281271709388 * log10Mh - 0.9477721865885471
    )  # T. Okabe 2020 Table 3
    se = 0.13  # T. Okabe 2020 Figure 9
    n = len(Mh)
    elip_fit[elip_fit < 0.233] = 0.233
    elip = st.truncnorm.rvs(
        (0.0 - elip_fit) / se, (0.9 - elip_fit) / se, loc=elip_fit, scale=se, size=n
    )
    return elip


def calc_vol(z, cosmo_col):
    """Volume appeared in cosmological 3D integral.

    :param z: redshift
    :type z: float
    :param cosmo_col: An instance of a colossus cosmology model
    :type cosmo_col: colossus.cosmology instance
    :return: volume
    :rtype: float
    """
    dis = cosmo_col.angularDiameterDistance(z) / (cosmo_col.H0 / 100.0)
    drdz = (2997.92458 / ((1.0 + z) * cosmo_col.Ez(z))) / (cosmo_col.H0 / 100.0)

    # 3282.806350011744 is 1rad^2 = 3282.806350011744[deg^2]
    # multiply fov[deg^2] to obtain the expected number to be observed after by KA
    return (dis * dis / 3282.806350011744) * drdz * (1.0 + z) * (1.0 + z) * (1.0 + z)


def dNhalodzdlnM_lens(M, z, cosmo_col):
    """Compute the differential number density of halos with respect to
    redshift and log halo mass, per a unit of solid angle [deg^2]

    :param M: The masses of the dark matter halos.
    :type  M: ndarray, float
    :param z: The redshift at which to compute the differential number
        density.
    :type  z: float
    :param cosmo_col: An instance of an colossus cosmology model
    :type cosmo_col: colossus.cosmology instance
    :return: dNhalodzdlnM, The differential number density of halos per
        unit redshift per natural log mass interval per unit area in
        units of #/deg^2/dlnM[M_sol/h].
    """
    dvoldzdO = calc_vol(z, cosmo_col)
    hhh = (cosmo_col.H0 / 100.0) ** 3
    mfunc_so = (
        mass_function.massFunction(M, z, mdef="fof", model="sheth99", q_out="dndlnM")
        * hhh
    )
    return dvoldzdO * mfunc_so


def concent_m_w_scatter(m, z, sig):
    """Concentration parameter of halos in B. Diemer and A. V. Kravtsov, 2015
    with updated parameters of  Diemer & Joyce 2019 arXiv:1407.4730 [astro-
    ph.CO]. arXiv:1809.07326 [astro-ph.CO].

    :param m: halo mass
    :type  m: nd.array
    :param z: redshift
    :type  z: float
    :param sig: intrinsic scatter of logarithmic concentration parameter
    :type  sig: float
    :return: con_halo: ndarray, concentration parameter of halos
    """
    con_mean = concentration.concentration(m, "200c", z, model="diemer19")
    sca = np.random.lognormal(0.0, sig, len(m))
    con = con_mean * sca
    con[con < 1.0] = 1.0  # TODO check
    return con
