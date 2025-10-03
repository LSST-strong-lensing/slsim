import numpy as np
import pytest
from colossus.cosmology import cosmology
from scipy import stats

from slsim.Deflectors.MassLightConnection.galaxy_population import galaxy_size
from slsim.Deflectors.MassLightConnection.galaxy_population import gals_init
from slsim.Deflectors.MassLightConnection.galaxy_population import modelscLognormal
from slsim.Deflectors.MassLightConnection.galaxy_population import set_gals_param
from slsim.Deflectors.MassLightConnection.galaxy_population import stellarmass_halomass


def test_galaxy_size():
    # Test the mean galaxy_size with valid arguments to ensure correct behavior for vdW23
    cosmo = cosmology.setCosmology("planck18")
    hubble = cosmo.H0 / 100.0
    mh = 1e12  # Example dark matter halo mass in units of Msun/h
    mstar = 10**10.8 * hubble  # Example stellar mass in units of Msun/h
    z = 0.75  # Example redshift
    q_out = "rb"  # Output query in units of Mpc/h
    model = "vdW23"  # Model to use

    # Verify the correct result is obtained
    expected_approx_size = (
        10**0.21 * hubble / 1e3 * 0.551
    )  # from van der Wel et al. 2023 Fig. 10 in units of Mpc/h
    calculated_size = galaxy_size(
        mh, mstar, z, cosmo, q_out, model, scatter=False
    )  #  in units of Mpc/h

    assert (
        pytest.approx(calculated_size, rel=0.1) == expected_approx_size
    ), "The galaxy size should match the expected value."

    # Verify the correct scatter is obtained
    n_ar = 1000
    mh_ar = np.array([1e12] * n_ar)
    calculated_size_w_scatters = galaxy_size(
        mh_ar, mstar, z, cosmo, q_out, model, scatter=True
    )
    logscatters = np.log(calculated_size_w_scatters / calculated_size)
    # Perform the KS test to see if the logged data follows a normal distribution
    D, p_value = stats.kstest(
        logscatters, "norm", args=(logscatters.mean(), logscatters.std(ddof=1))
    )
    # Set the significance level at which you are testing
    alpha = 0.05  # Commonly used significance level
    assert (
        p_value > alpha
    ), f"Sample data does not follow a lognormal distribution (p-value={p_value})"

    # test_galaxy_size_unknown_model
    with pytest.raises(Exception) as excinfo:
        galaxy_size(1e12, 1e11, 0.3, cosmo, model="unknown_model")
    assert "Unknown model" in str(
        excinfo.value
    ), "An exception with 'Unknown model' message should be raised for unknown models."

    # test_galaxy_size_unknown_model
    with pytest.raises(Exception) as excinfo:
        galaxy_size(1e12, 1e11, 0.3, cosmo, q_out="unknown_output")
    assert "Unknown output" in str(
        excinfo.value
    ), "An exception with 'Unknown model' message should be raised for unknown output"

    # test_mh_is_not_float/list/nd.array_case
    with pytest.raises(Exception) as excinfo:
        galaxy_size("mass_h", 1e11, 0.3, cosmo)
    assert "type(mh) should be float, ndarray or list." in str(
        excinfo.value
    ), "An exception with 'type(mh)' message should be raised for mh is not float/list/nd.array"

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for orugi20 model
    model2 = "oguri20"  # Model to use
    sig_tb = 0.2
    calculated_size_model2 = galaxy_size(
        mh, mstar, z, cosmo, q_out, model2, scatter=True, sig_tb=sig_tb
    )  # in units of Mpc/h
    assert (
        0 < calculated_size_model2 < 0.1
    ), "The galaxy size should be less than 100 kpc"

    # Test the mean galaxy_size with valid arguments to ensure correct behavior for karmakar23 model
    model3 = "karmakar23"
    calculated_size_model3 = galaxy_size(
        mh, mstar, z, cosmo, q_out, model3, scatter=True
    )  # in units of Mpc/h
    assert (
        0 < calculated_size_model3 < 0.1
    ), "The galaxy size should be less than 100 kpc"


def test_modelscLognormal():
    lnsig = 0.1
    n = 1
    random = modelscLognormal(lnsig, n)
    assert 0 < random < np.exp(lnsig * 5)


def test_stellarmass_halomass():
    mh_ar = np.logspace(10, 16, 100)  # in units of M_sol/h
    z = 0.5
    # TYPE_SMHM="true"
    paramc, params = gals_init()
    stellar_masses = stellarmass_halomass(mh_ar, z, paramc)  # in units of M_sol/h
    ratio_SMHM = stellar_masses / mh_ar
    index_of_max = np.argmax(ratio_SMHM)
    Mh_at_peak_ratio_SMHM = mh_ar[index_of_max]
    assert (
        pytest.approx(np.log10(Mh_at_peak_ratio_SMHM), rel=0.02) == 12
    ), "The SMHM ratio should be peaked at Mh ~ 1e12 M_sol"
    assert all(
        0 <= r <= 0.1 for r in ratio_SMHM
    ), "The SMHM conversion rate should be at most 10%"

    # TYPE_SMHM="obs"
    paramc, params = gals_init(TYPE_SMHM="obs")
    stellar_masses = stellarmass_halomass(mh_ar, z, paramc)  # in units of M_sol/h
    ratio_SMHM = stellar_masses / mh_ar
    index_of_max = np.argmax(ratio_SMHM)
    Mh_at_peak_ratio_SMHM = mh_ar[index_of_max]
    assert (
        pytest.approx(np.log10(Mh_at_peak_ratio_SMHM), rel=0.02) == 12
    ), "The SMHM ratio should be peaked at Mh ~ 1e12 M_sol"
    assert all(
        0 <= r <= 0.1 for r in ratio_SMHM
    ), "The SMHM conversion rate should be at most 10%"

    # TYPE_SMHM="true_all"
    paramc, params = gals_init(TYPE_SMHM="true_all")
    stellar_masses = stellarmass_halomass(mh_ar, z, paramc)  # in units of M_sol/h
    ratio_SMHM = stellar_masses / mh_ar
    index_of_max = np.argmax(ratio_SMHM)
    Mh_at_peak_ratio_SMHM = mh_ar[index_of_max]
    assert (
        pytest.approx(np.log10(Mh_at_peak_ratio_SMHM), rel=0.02) == 12
    ), "The SMHM ratio should be peaked at Mh ~ 1e12 M_sol"
    assert all(
        0 <= r <= 0.1 for r in ratio_SMHM
    ), "The SMHM conversion rate should be at most 10%"


def test_set_gals_param():
    p_h = np.zeros(1000)  # in units of degree
    e_gal, p_gal = set_gals_param(p_h)  # in units of degree
    assert all(0 <= elipticily <= 1 for elipticily in e_gal)
    assert all(-180 <= p_angle <= 180 for p_angle in p_gal)
    sigma_p_gal = np.std(p_gal)
    expected_alignment = 35.4  # in units of degree
    assert pytest.approx(sigma_p_gal, rel=0.1) == expected_alignment
