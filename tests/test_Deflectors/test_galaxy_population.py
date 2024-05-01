import numpy as np
import pytest
from colossus.cosmology import cosmology
from slsim.Deflectors.galaxy_population import(
    galaxy_size,
    gene_e,
    gene_ang_gal,
    gals_init,
    stellarmass_halomass
)
from scipy import stats


def test_galaxy_size():
    # Test the mean galaxy_size with valid arguments to ensure correct behavior for vdW23
    cosmo = cosmology.setCosmology("planck18")
    hubble = cosmo.H0/100.
    mh = 1e12  # Example dark matter halo mass in units of Msun/h
    mstar = 1e10*hubble  # Example stellar mass in units of Msun/h
    z = 0.75  # Example redshift
    q_out = "rb"  # Output query in units of Mpc/h
    model = "vdW23"  # Model to use

    # Verify the correct result is obtained
    expected_approx_size = 10**0.125*hubble/1e3*0.551  # from van der Wel et al. 2023 Fig. 10 in units of Mpc/h
    calculated_size = galaxy_size(mh, mstar, z, cosmo, q_out, model,scatter=False)#  in units of Mpc/h

    assert pytest.approx(calculated_size, rel = 0.1) == expected_approx_size, "The galaxy size should match the expected value."

    # Verify the correct scatter is obtained
    n_ar = 1000
    mh_ar = np.array([1e12]*n_ar)
    calculated_size_w_scatters = galaxy_size(mh_ar, mstar, z, cosmo, q_out, model,scatter=True)
    logscatters = np.log(calculated_size_w_scatters/calculated_size)
    # Perform the KS test to see if the logged data follows a normal distribution
    D, p_value = stats.kstest(logscatters, 'norm', args=(logscatters.mean(), logscatters.std(ddof=1)))
    # Set the significance level at which you are testing
    alpha = 0.05  # Commonly used significance level
    assert p_value > alpha, f"Sample data does not follow a lognormal distribution (p-value={p_value})"

    #test_galaxy_size_unknown_model
    with pytest.raises(Exception) as excinfo:
        galaxy_size(1e12, 1e11, 0.3, cosmo, model="unknown_model")
    assert "Unknown model" in str(excinfo.value), "An exception with 'Unknown model' message should be raised for unknown models."

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for orugi20 model
    model2 = "oguri20"  # Model to use
    sig_tb = 0.2
    calculated_size_model2 = galaxy_size(mh, mstar, z, cosmo, q_out, model2,scatter=False, sig_tb=sig_tb) #in units of Mpc/h
    assert 0< calculated_size_model2 <0.01, "The galaxy size should be less than 10 kpc"

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for karmakar23 model
    model3 = "karmakar23"
    calculated_size_model3 = galaxy_size(mh, mstar, z, cosmo, q_out, model3,scatter=False) #in units of Mpc/h
    assert 0< calculated_size_model3 <0.01, "The galaxy size should be less than 10 kpc"



def test_stellarmass_halomass():
    mh_ar = np.logspace(10,16, 100) # in units of M_sol/h
    z = 0.5
    paramc, params = gals_init()
    stellar_masses = stellarmass_halomass(mh_ar, z, paramc) # in units of M_sol/h
    ratio_SMHM = stellar_masses/mh_ar
    index_of_max = np.argmax(ratio_SMHM)
    Mh_at_peak_ratio_SMHM = mh_ar[index_of_max]
    assert pytest.approx(np.log10(Mh_at_peak_ratio_SMHM), rel = 0.02) == 12, "The SMHM ratio should be peaked at Mh ~ 1e12 M_sol"
    assert all(0 <= r <= 0.1 for r in ratio_SMHM), "The SMHM conversion rate should be at most 10%"

def test_gene_e():
    n = 10
    e_gal = gene_e(n)
    assert all(0 <= elipticily <= 1 for elipticily in e_gal)


def test_gene_ang_gal():
    p_h = np.zeros(1000) # in units of degree
    p_gal  = gene_ang_gal(p_h) # in units of degree
    assert all(-180 <= p_angle <= 180 for p_angle in p_gal)
    sigma_p_gal = np.std(p_gal)
    expected_alignment = 35.4 # in units of degree
    assert pytest.approx(sigma_p_gal, rel = 0.1) ==  expected_alignment


