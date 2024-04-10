import numpy as np
import glafic

from scipy import optimize
from scipy import interpolate
from colossus.lss import mass_function
from colossus.halo import concentration
from colossus.halo import mass_so

import global_value as g
import gen_mock_halo
import lens_subhalo
import lens_gals
import source_tab
#
# number counts and properties of lens halo
#


def dNhalodzdlnM_lens(M, z, cosmo):
    dvoldzdO = source_tab.calc_vol(z, cosmo)
    hhh = (cosmo.H0 / 100.0)**3
    mfunc_so = mass_function.massFunction(
        M, z, mdef='fof', model='sheth99', q_out='dndlnM')*hhh
    return dvoldzdO*mfunc_so  # [number/deg^2/dlnM[Modot/h]]


def concent_m(m, z):
    return concentration.concentration(m, 'vir', z, model='diemer15')


def concent_m_w_scatter(m, z, sig):
    con = concentration.concentration(m, 'vir', z, model='diemer15')
    sca = np.random.lognormal(0.0, sig, len(m))
    return con*sca


def critical_surface_density(zl, zs, cosmo):
    rl = cosmo.comovingDistance(z_max=zl)
    rs = cosmo.comovingDistance(z_max=zs)
    dol = (1.0 / (1.0 + zl)) * rl
    dos = (1.0 / (1.0 + zs)) * rs
    dls = (1.0 / (1.0 + zs)) * (rs - rl)
    sig_cr = g.c2_G / 4./np.pi*dos/dol/dls
    return sig_cr


def bnorm_hern(Mste, rb, zl, zs, cosmo):
    s_cr = critical_surface_density(zl, zs, cosmo)
    b_norm = Mste/2./np.pi/rb**2/s_cr
    return b_norm


def mtot_nfw(c):
    return np.log(1.0 + c) - (c / (1.0 + c))


def F_nfw(uu):
    if uu < 1:
        F = (1./(np.sqrt(1.-uu**(2))))*np.arctanh(np.sqrt(1.-uu**(2)))
    else:
        F = (1./(np.sqrt(uu**(2)-1.)))*np.arctan(np.sqrt(uu**(2)-1.))
    return F


def kappa_dl_nfw(uu):
    F_u = F_nfw(uu)
    kappa_dl = 1./(2*(uu**(2.)-1.))*(1.-F_u)
    return kappa_dl


def kappa_dl_hern(uu):
    F_u = F_nfw(uu)
    kappa_dl = 1./((uu**(2.)-1.)**(2.))*(-3.+(2.+uu**(2.))*F_u)
    return kappa_dl


def bnorm_nfw(Mhalo, con, zl, zs, cosmo):
    s_cr = critical_surface_density(zl, zs, cosmo)
    rvir = 1.0e-3 * mass_so.M_to_R(Mhalo, zl, 'vir')
    rs = rvir/con
    rhos = mass_so.deltaVir(zl)*(cosmo.rho_c(zl) /
                                 g.kpc_to_Mpc**3)*con**3/3./mtot_nfw(con)
    return 4*rs*rhos/s_cr


def create_interpolator3(A_values, B_values, C_values, precomputed_grid):
    interpolator = interpolate.RegularGridInterpolator(
        (A_values, B_values, C_values), precomputed_grid)
    return interpolator


def func_mag(x, y):
    img = glafic.calcimage(g.zsmax, x, y, verb=0)
    return 1.0 / np.abs(img[6])


def func_magx_root(x, mag):
    return func_mag(x, 0.0) - mag


def func_magy_root(y, mag):
    return func_mag(0.0, y) - mag


def func_kap(x, y):
    img = glafic.calcimage(g.zsmax, x, y, verb=0)
    return img[3]


def func_kapx_root(x, mag):
    return func_kap(x, 0.0) - mag


def func_kapy_root(y, mag):
    return func_kap(0.0, y) - mag


def create_interp_bsrc_h(zmin, zmax, Mhmin, Mhmax, cosmo):

    mmh = np.logspace(np.log10(Mhmin), np.log10(Mhmax), 50)
    zz_te = np.linspace(zmin, zmax, 50)

    # derive boundary box size in the source plane
    eh = 0.8
    ph = 0.0
    es = 0.8
    ps = 0.0

    src_y_ar = np.zeros((len(mmh), len(zz_te)))

    for i, mh in enumerate(mmh):
        for k, z in enumerate(zz_te):
            mcen = lens_gals.stellarmass_halomass(
                mh/(cosmo.H0/100.), z, g.paramc, g.frac_SM_IMF)*10**(g.sig_mcen)*(cosmo.H0/100.)
            c = concent_m(mh, z)*10**(g.sig_c)
            tb = lens_gals.galaxy_size(
                mh, mcen/g.frac_SM_IMF,  z, cosmo, model=g.TYPE_GAL_SIZE)
            comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(
                cosmo)
            glafic.init(comega, clambda, cweos, chubble,
                        'out2', -20.0, -20.0, 20.0, 20.0, 0.2, 0.2, 5, verb=0)

            glafic.startup_setnum(2, 0, 0)
            glafic.set_lens(1, 'anfw',  z, mh, 0.0, 0.0, eh, ph, c,  0.0)
            glafic.set_lens(2, 'ahern', z, mcen, 0.0, 0.0, es, ps, tb, 0.0)

            # model_init needs to be done again whenever model parameters are changed
            glafic.model_init(verb=0)

            kap_th = 0.45

            zz = optimize.root_scalar(
                func_kapy_root, method='brentq', args=(kap_th), bracket=(1.0e-4, 2000.0))
            kap_y = zz.root

            glafic.calcimage(g.zsmax, 0.0, kap_y, verb=0)

            mag_th = 1.5

            zz = optimize.root_scalar(
                func_magy_root, method='brentq', args=(mag_th), bracket=(kap_y, 1.0e6))
            box_y = zz.root

            img = glafic.calcimage(g.zsmax, 0.0, box_y, verb=0)
            src_y = box_y - img[1]
            src_y_ar[i, k] = src_y
            glafic.quit()
    log10mmh = np.log10(mmh)
    interp_bsrc_h = lens_subhalo.create_interpolator(log10mmh, zz_te, src_y_ar)
    np.savez('result/' + g.prefix + '_interp_bsrc_h.npz',
             log10mmh, zz_te, src_y_ar)
    return interp_bsrc_h


#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_halo.init_cosmo()
    g.sig_c = 0.13
    g.sig_mcen = 0.3
    g.paramc, g.params = lens_gals.gals_init()
    g.prefix = "check"
    interp_bsrc_h = create_interp_bsrc_h(0.1, 3.0, 1e11, 3e16, cosmo)
    test = [13.4, np.log10(concent_m(13.4, 1.2), 1.2)]
    print(interp_bsrc_h(test))
