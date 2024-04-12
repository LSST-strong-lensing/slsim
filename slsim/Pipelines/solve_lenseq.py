import numpy as np
import scipy.stats as st
# import solve_lenseq_lenstronomy


def gene_e(n):
    em = 0.3
    se = 0.16
    e = st.truncnorm.rvs((0.0 - em) / se, (0.9 - em) /
                         se, loc=em, scale=se, size=n)
    return e


def gene_e_ang_halo(Mh):
    n = len(Mh)
    e = gene_e_halo(Mh)
    p = gene_ang(n)
    return e, p


def gene_e_halo(Mh):
    log10Mh = np.log10(Mh)  # log10([Modot/h])
    elip_fit = 0.09427281271709388*log10Mh - \
        0.9477721865885471  # T. Okabe 2020 Table 3
    se = 0.13  # T. Okabe 2020 Figure 9
    n = len(Mh)
    elip_fit[elip_fit < 0.233] = 0.233
    elip = st.truncnorm.rvs(
        (0.0 - elip_fit) / se, (0.9 - elip_fit) / se, loc=elip_fit, scale=se, size=n)
    return elip


def gene_ang(n):
    return (np.random.rand(n) - 0.5) * 360.0


def gene_ang_gal(pol_h):
    n = len(pol_h)
    sig = 35.4  # Okumura + 2009
    pol_gal = np.random.normal(loc=pol_h, scale=sig,size=n)
    return pol_gal


def gene_gam(z):
    if z < 1.0:
        sig = 0.023 * z
    else:
        sig = 0.023 + 0.032 * np.log(z)

    g1 = st.truncnorm.rvs(-0.5 / sig, 0.5 / sig, loc=0.0, scale=sig)
    g2 = st.truncnorm.rvs(-0.5 / sig, 0.5 / sig, loc=0.0, scale=sig)

    return np.sqrt(g1 * g1 + g2 * g2)


def set_shear(z):
    gg = gene_gam(z)
    tgg = gene_ang(1)[0]
    return gg, tgg

