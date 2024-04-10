import numpy as np
import scipy.stats as st
import gen_mock_halo
import solve_lenseq_glafic
import global_value as g
# import solve_lenseq_lenstronomy

def calc_image(lens_par, srcs_par, ein, rt_range, flag_mag, cosmo):
    """
    This function calculates various parameters related to an image.

    Parameters:
    - lens_par (list): List containing lens parameters
    - srcs_par (list): List containing source parameters
    - ein: Value of Einstein radius in units of [arcsec]
    - rt_range: ratio that means the margin of the calculation area
    - flag_mag: Flag for magnitude
    - cosmo: Cosmological model in Colossus

    Returns:
    - out_img (ndarray): return of the ``point_solve" function in glafic
    - nim: Number of images
    - sep: maximum separation between images in units of [arcsec]
    - mag: magnitude of the first arrival image
    - mag_max: maximum magnitude
    - fr: Ratio of flux for 2-, 3- imgs lens
    - kapgam (list): convergence, external shear, and stellar convergence
    """

    out_img, kapgam = solve_lenseq_glafic.solve_lenseq_glafic(
        lens_par, srcs_par, ein, rt_range, cosmo)

    nim = len(out_img)

    if nim == 1:
        n = 0.0
        a = []
        return out_img, nim, n, out_img[0][2], out_img[0][2], n, kapgam
    if nim < 1:
        n = 0.0
        a = []
        return out_img, nim, n, n, n, n, a

    xi, yi, mi = [], [], []
    for i in range(nim):
        xi.append(out_img[i][0])
        yi.append(out_img[i][1])
        mi.append(abs(out_img[i][2]))

    mi.sort(reverse=True)
    mag_tot = sum(mi)
    mag_max = max(mi)

    si = []
    for i in range(nim - 1):
        for j in range(i + 1, nim):
            si.append(((xi[i] - xi[j]) ** 2) + ((yi[i] - yi[j]) ** 2))
    sep = np.sqrt(max(si))

    if flag_mag > 0:
        ii = min(flag_mag, nim-1)
        ii = max(ii, 2)
        mag = mi[ii - 1]
    else:
        mag = mag_tot

    fr = mi[1] / mi[0] if nim in [2, 3] else 1.0

    return out_img, nim, sep, mag, mag_max, fr, kapgam


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


#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_halo.init_cosmo()
    print(cosmo.Om0, round(1.0+g.nonflat-cosmo.Om0, 5),
          g.cosmo_weos, round(cosmo.H0/100., 5))
