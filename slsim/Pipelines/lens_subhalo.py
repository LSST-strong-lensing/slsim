import numpy as np
from scipy import optimize

import glafic
import lens_halo
import lens_gals
import gen_mock_halo
from colossus.lss import peaks
from colossus.halo import mass_so
import global_value as g
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy import integrate
from scipy import optimize
#
# number counts and properties of lens subhalo
#

fac_f = 0.5


def deltac_z(z, cosmo):
    return (3.0 / 20.0) * ((12.0 * np.pi) ** (2.0 / 3.0)) * (cosmo.Om(z) ** 0.0055) / cosmo.growthFactor(z)


def zf_func(zf, dz, s1, s2, cosmo):
    return dz - deltac_z(zf, cosmo) + 0.974 * np.sqrt((s1 * s1 - s2 * s2) / 0.707)


def zf_calc(z, m, cosmo):
    s1 = cosmo.sigma(peaks.lagrangianR(fac_f * m))
    s2 = cosmo.sigma(peaks.lagrangianR(m))
    dz = deltac_z(z, cosmo)
    z = optimize.root_scalar(zf_func, args=(
        dz, s1, s2, cosmo), method='brentq', bracket=(z, 100.0))

    return z.root


def mtot_tnfw2(t):
    t2 = t * t + 1.0
    ff = t * t / (2.0 * t2 * t2 * t2)
    gg = 2.0 * t * t * (t * t - 3.0) * np.log(t) - \
        (3.0 * t * t - 1.0) * (t * t + 1.0 - np.pi * t)

    return ff * gg


def mf_sub_eps(m, z, mh, zf, cosmo):
    dc1 = deltac_z(zf, cosmo)
    dc2 = deltac_z(z, cosmo)

    s1 = cosmo.sigma(peaks.lagrangianR(m))
    s2 = cosmo.sigma(peaks.lagrangianR(mh))

    h = 0.005
    dsdm1 = (cosmo.sigma(peaks.lagrangianR(m * (1.0 - h))) -
             cosmo.sigma(peaks.lagrangianR(m * (1.0 + h)))) / (2.0 * m * h)

    x = (dc1 - dc2) / np.sqrt(2.0 * np.abs(s1 * s1 - s2 * s2))
    f = (x / np.sqrt(np.pi)) * np.exp(-(1.0)
                                      * x * x) / np.abs(s1 * s1 - s2 * s2)
    dsdm2 = 2.0 * s1 * dsdm1

    mf = f * (mh / m) * dsdm2

    if s1.size == 1:
        if s1 < s2:
            mf = 0.0
    else:
        mf[s1 < s2] = 0.0

    return mf


def sub_m_func(m, mt, zf, rt, cosmo):
    # Before tidal effect, this subhalo should be adopted concentration parameter of field halo, Yes
    c = lens_halo.concent_m(m, zf)
    rs = 1.0e-3 * mass_so.M_to_R(m, zf, 'vir') / c

    return m * mtot_tnfw2(rt / rs) / lens_halo.mtot_nfw(c) - mt


def t_df(mh, m, z, cosmo):
    hubble = cosmo.H0 * 1.0e-2
    rvir = 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') / hubble
    msc = (1476.6697 * (mh / hubble) / 3.085677581e16) * 1.0e-6
    vvir = (np.sqrt(msc / rvir) * (2.99792458e8) / 3.085677581e16) * 1.0e-6
    tdyn = ((rvir / vvir) / (3.154e7)) * 1.0e-9

    return 2.0 * (mh / m) * tdyn


def fac_mf_sub(mh, m, z, zf, cosmo):
    dt = cosmo.age(z) - cosmo.age(zf)
    x = dt / t_df(fac_f * mh, m, zf, cosmo)

    return np.exp((-1.0) * x * x)


def rt_calc(mh, m, z, cosmo):
    c = lens_halo.concent_m(mh, z)
    rs = 1.0e-3 * mass_so.M_to_R(mh, z, 'vir') / c
    lx1 = np.log(1.0e-6)
    lx2 = np.log(1.0e4)

    tau = tnfw2_tcalc(c)

    integ, err = integrate.quad(rt_sub_ave_func, lx1, lx2, args=(mh, tau))

    return rs * integ * (m ** (1.0 / 3.0))


def sub_m_calc(mh, mt, z, zf, spl, cosmo):
    rt = rt_calc(fac_f * mh, mt, zf, cosmo)
    mo = 10.0 ** spl(np.log10(mt))

    return mo, rt


def sub_m_calc_setspline(mh, m, z, zf, cosmo):
    # Get the representative value of "m", i.e. the subhalo mass "after" tidal stripping
    # Then creating array for the subhalo mass array surrounding the mass "after" tidal stripping
    m1 = np.min(m) * (10.0 ** (-0.3))
    m2 = np.max(m) * (10.0 ** 0.301)
    logm = np.arange(np.log10(m1), np.log10(m2), 0.1)

    # Calculating the averaged truncation radius for the subhalos from their mass "after" tidal stripping
    rt = rt_calc(fac_f * mh, 10.0 ** logm, zf, cosmo)

    # "mo" means the subhalo mass before tidal stripping
    logmo = np.zeros_like(logm)
    for i in range(len(logm)):
        # the subhalo mass "after" tidal stripping
        mt = 10.0 ** logm[i]
        # Sampling the subhalo mass "before" tidal stripping for the various value of the subhalo mass "after" tidal stripping
        z = optimize.root_scalar(sub_m_func, args=(
            mt, zf, rt[i], cosmo), method='brentq', bracket=(0.5 * mt, 100.0 * mt))
        # Solving the subhalo mass "before" tidal stripping
        logmo[i] = np.log10(z.root)

    # Spline function of the "before" mass from the "after" mass
    logmo_spline = _spline(logm, logmo, k=5)

    return logmo_spline


def exnum_sh_oguri_w_macc_for_grid(Mh, z, cosmo, min_Msh=1.e10, n_bins=100):
    mmsh = np.logspace(np.log10(min_Msh), np.log10(Mh/2.), n_bins)
    zf = zf_calc(z, Mh, cosmo)  # KA
    spl = sub_m_calc_setspline(Mh, mmsh, z, zf, cosmo)
    mo, rt = sub_m_calc(Mh, mmsh, z, zf, spl, cosmo)

    h = 0.02
    mop, rtp = sub_m_calc(Mh, mmsh * (1.0 + h), z, zf, spl, cosmo)
    mom, rtm = sub_m_calc(Mh, mmsh * (1.0 - h), z, zf, spl, cosmo)
    dmodm = (mop - mom) / (2.0 * h * mmsh)
    dndmo = mf_sub_eps(mo, z, Mh, zf, cosmo)
    fac = fac_mf_sub(Mh, mo, z, zf, cosmo)
    dndlogmsh = mmsh * dndmo * dmodm * fac
    dlogmsh = np.diff(np.log(mmsh), n=1)
    dndlogmsh = dndlogmsh[1::]
    dnsh = dndlogmsh*dlogmsh
    return mmsh[1::]/Mh, mo[1::]/Mh, dnsh


def concent_m_sub_ando(m, z, cosmo):
    ez = cosmo.Ez(z)
    hubble = cosmo.H0 * 1.0e-2

    lm = np.log(m / hubble)
    c200 = 94.6609 + lm * (-4.1160 + lm * (0.033747 + lm * (2.0932e-4)))
    if c200.size == 1:
        if c200 < 0.1:
            c200 = 0.1
    else:
        c200[c200 < 0.1] = 0.1

    deltaomega = mass_so.deltaVir(z) * (ez * ez)
    fac = (200.0 / (deltaomega / (ez * ez))) ** (1.0 / 3.0)

    return fac * c200 / (ez ** (2.0/3.0))


def tnfw2_tcalc_func(tau_h, c_h):
    return mtot_tnfw2(tau_h) - lens_halo.mtot_nfw(c_h)


def tnfw2_tcalc(c_h):
    z = optimize.root_scalar(tnfw2_tcalc_func, args=(
        c_h), method='brentq', bracket=(0.5 * c_h, 10.0 * c_h))
    return z.root


def pr_bmo(x, tau_h):
    return (x * x / ((1.0 + x) * (1.0 + x))) * (tau_h * tau_h / (x * x + tau_h * tau_h)) * (tau_h * tau_h / (x * x + tau_h * tau_h)) / mtot_tnfw2(tau_h)

# Dimensionless total mass inside the arbitrary radius with truncated NFW(BMO) profile
# To convert to the dimensional mass, multiply 4pi\rhos rs^3, so that M(<x*rs)=4pi\rhos rs^3* tnfw2_m3d(x,c,t)


def tnfw2_m3d(x, tau_h):
    tau_h2 = tau_h * tau_h + 1.0

    ff = tau_h * tau_h / (2.0 * tau_h2 * tau_h2 * tau_h2 *
                          (1.0 + x) * (tau_h * tau_h + x * x))
    gg = tau_h2 * x * (x * (x + 1.0) - tau_h * tau_h * (x - 1.0) * (2.0 + 3.0 * x) - 2.0 * tau_h * tau_h * tau_h * tau_h) + tau_h * (x + 1.0) * (tau_h * tau_h + x * x) * (
        2.0 * (3.0 * tau_h * tau_h - 1.0) * np.arctan(x / tau_h) + tau_h * (tau_h * tau_h - 3.0) * np.log(tau_h * tau_h * (1.0 + x) * (1.0 + x) / (tau_h * tau_h + x * x)))

    return ff * gg

# Calculation of the average truncation radius for subhalos following Eq.(B6) of Oguri&Takahashi 2020


def rt_sub_nom(x, mh, tau_h):
    # M(<x) for host halo
    menc = mh * tnfw2_m3d(x, tau_h) / mtot_tnfw2(tau_h)

    return x * (1.0 / (3.0 * menc)) ** (1.0 / 3.0)


def rt_sub_ave_func(lx, mh, tau_h):
    x = np.exp(lx)
    # (4*pi*rs^3)*U(r|M,m) * x(1/3M(<r))^1/3
    return pr_bmo(x,  tau_h) * rt_sub_nom(x, mh, tau_h)


def random_points_on_elip_2d(r, e, p, num_points):
    z_pre = np.random.rand(num_points)
    z = (z_pre-0.5)*2.0
    phi = np.random.rand(num_points)*2.*np.pi
    sqrt_1_minus_z2 = np.sqrt(1. - z**2)
    x = sqrt_1_minus_z2*np.cos(phi)*r*np.sqrt(1.-e)
    y = sqrt_1_minus_z2*np.sin(phi)*r/np.sqrt(1.-e)
    pp = p * np.pi / 180.0
    xelip = x*np.cos(pp) - y*np.sin(pp)
    yelip = x*np.sin(pp) + y*np.cos(pp)
    return xelip, yelip


def subhalo_distribute(rvir_h, con_h, e_h, p_h, xfunc, n):
    fnfw = lens_halo.mtot_nfw(con_h)
    u = np.random.rand(n)
    ufnwf = u*fnfw
    x_sol = xfunc(ufnwf)
    x_sol[x_sol < 0.0001] = 0.0001
    radius_sh = rvir_h/con_h*x_sol  # in physical[h^-1 Mpc]
    x_sh_elip, y_sh_elip = random_points_on_elip_2d(
        radius_sh, e_h, p_h, n)
    return x_sh_elip, y_sh_elip


def precompute_dnsh(Mh_values, zz_values, output_length, cosmo, min_Msh=1e10):
    grid_msh_acc_Mh = np.zeros((len(Mh_values), len(zz_values), output_length))
    grid_dnsh = np.zeros((len(Mh_values), len(zz_values), output_length))
    for i, mh in enumerate(Mh_values):
        for j, z in enumerate(zz_values):
            msh_Mh, grid_msh_acc_Mh[i, j], grid_dnsh[i, j] = exnum_sh_oguri_w_macc_for_grid(
                mh, z, cosmo, min_Msh=min_Msh, n_bins=output_length+1)

    return grid_msh_acc_Mh, grid_dnsh


def create_interpolator(A_values, B_values, precomputed_grid):
    interpolator = interpolate.RegularGridInterpolator(
        (A_values, B_values), precomputed_grid)
    return interpolator


def create_interp_bsrc_sh(zmin, zmax, Mshmin, Mshmax, cosmo):

    # this mmsh means mass of subhalos at their accretion time
    mmsh = np.logspace(np.log10(Mshmin), np.log10(Mshmax), 50)
    zz_te = np.linspace(zmin, zmax, 50)

    # derive boundary box size in the source plane
    eh = 0.8
    ph = 0.0
    es = 0.8
    ps = 0.0

    src_y_ar = np.zeros((len(mmsh), len(zz_te)))

    for i, msh in enumerate(mmsh):
        for k, z_l in enumerate(zz_te):

            c_sh = max(concent_m_sub_ando(msh, z_l, cosmo),
                       lens_halo.concent_m(msh, z_l))*10**g.sig_c_sh
            msat = lens_gals.stellarmass_halomass(
                msh/(cosmo.H0/100.), z_l, g.params, g.frac_SM_IMF)*10**(g.sig_msat)*(cosmo.H0/100.)

            tb = lens_gals.galaxy_size(
                msh, msat/g.frac_SM_IMF, z_l, cosmo, model=g.TYPE_GAL_SIZE)
            comega, clambda, cweos, chubble = gen_mock_halo.calc_cosmo_for_glafic(
                cosmo)
            glafic.init(comega, clambda, cweos, chubble,
                        'out2', -20.0, -20.0, 20.0, 20.0, 0.2, 0.2, 5, verb=0)

            glafic.startup_setnum(2, 0, 0)
            glafic.set_lens(1, 'anfw',  z_l, msh, 0.0, 0.0, eh, ph, c_sh,  0.0)
            glafic.set_lens(2, 'ahern', z_l, msat, 0.0, 0.0, es, ps, tb, 0.0)

            # model_init needs to be done again whenever model parameters are changed
            glafic.model_init(verb=0)

            kap_th = 0.45

            zz = optimize.root_scalar(
                lens_halo.func_kapy_root, method='brentq', args=(kap_th), bracket=(1.0e-4, 2000.0))
            kap_y = zz.root

            glafic.calcimage(g.zsmax, 0.0, kap_y, verb=0)

            mag_th = 1.5

            zz = optimize.root_scalar(
                lens_halo.func_magy_root, method='brentq', args=(mag_th), bracket=(kap_y, 1.0e6))
            box_y = zz.root

            img = glafic.calcimage(g.zsmax, 0.0, box_y, verb=0)
            src_y = box_y - img[1]
            src_y_ar[i, k] = src_y
            glafic.quit()

    log10mmsh = np.log10(mmsh)
    interp_bsrc_sh = create_interpolator(log10mmsh, zz_te, src_y_ar)
    np.savez('result/' + g.prefix + '_interp_bsrc_sh.npz',
             log10mmsh, zz_te, src_y_ar)
    return interp_bsrc_sh


def create_interp_dndmsh(zmin, zmax, Mhmin, Mhmax, Msh_min, cosmo, n_bins=100):
    zz_int_comp = np.linspace(zmin, zmax, 30)
    log10MMh_int_comp = np.linspace(np.log10(Mhmin/2.), np.log10(Mhmax), 30)
    output_length = n_bins-1

    # Precompute subhalo mass function & Create the interpolator
    grid_msh_acc_Mhp, grid_dnshp = precompute_dnsh(
        10**log10MMh_int_comp, zz_int_comp, output_length, cosmo, min_Msh=Msh_min)
    interp_dnsh = create_interpolator(
        log10MMh_int_comp, zz_int_comp, grid_dnshp)
    interp_msh_acc_Mh = create_interpolator(
        log10MMh_int_comp, zz_int_comp, grid_msh_acc_Mhp)
    np.savez('result/' + g.prefix + '_interp_dnsh_msh_acc_Mh.npz', log10MMh_int_comp,
             zz_int_comp, grid_dnshp, grid_msh_acc_Mhp)
    return interp_dnsh, interp_msh_acc_Mh


#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_halo.init_cosmo()
    g.sig_c_sh = 0.13
    g.sig_msat = 0.3
    g.paramc, g.params = lens_gals.gals_init()
    g.prefix = "check"
    g.TYPE_GAL_SIZE = 'oguri20'
    interp_bsrc_sh = create_interp_bsrc_sh(0.1, 3.0, 1e10, 2e15, cosmo)
    test = [13.4, 1.2]
    print(interp_bsrc_sh(test))
