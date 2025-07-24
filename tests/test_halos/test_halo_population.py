import numpy as np
from colossus.cosmology import cosmology
from slsim.Halos.halo_population import (
    gene_e_ang_halo,
    calc_vol,
    dNhalodzdlnM_lens,
    concent_m_w_scatter,
)

# Assuming other imports are already defined, we continue from here.


def test_gene_e_ang_halo():
    Mh_ar = np.logspace(10, 16, 10)  # in units of M_sol/h
    e_h, p_h = gene_e_ang_halo(Mh_ar)
    assert all(0 <= elipticily <= 1 for elipticily in e_h)
    assert all(-180 <= p_angle <= 180 for p_angle in p_h)


def test_calc_vol():
    z_ar = np.linspace(0, 5, 10)
    cosmo = cosmology.setCosmology("planck18")
    vol_ar = calc_vol(z_ar, cosmo)
    assert all(vol > 0 for vol in vol_ar)


def test_dNhalodzdlnM_lens():
    Mh_ar = np.logspace(10, 16, 10)  # in units of M_sol/h
    z_ar = np.linspace(0, 5, 10)
    cosmo = cosmology.setCosmology("planck18")
    for z in z_ar:
        dN_ar = dNhalodzdlnM_lens(Mh_ar, z, cosmo)
        assert all(num > 0 for num in dN_ar)


def test_concent_m_w_scatter():
    Mh_ar = np.logspace(10, 16, 10)  # in units of M_sol/h
    z_ar = np.linspace(0, 5, 5)
    lnsigma = 0.3
    # cosmo = cosmology.setCosmology("planck18")
    for z in z_ar:
        con_ar = concent_m_w_scatter(Mh_ar, z, lnsigma)
        assert all(con >= 1 for con in con_ar)
