import pytest
import numpy as np
from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen.qsosed import (
    Quasar_sed,
    pl,
    bb,
    tau_eff,
    four_pi_dL_sq,
)
from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen.config import params_agile
from astropy.cosmology import FlatLambdaCDM

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def cosmo():
    """Returns a standard flat LambdaCDM cosmology for testing."""
    return FlatLambdaCDM(H0=70.0, Om0=0.3)


@pytest.fixture
def mock_wavelengths():
    """Create a standard wavelength array for testing.

    Must cover 4000-5000A (rest frame) to satisfy host galaxy
    normalization checks.
    """
    return np.linspace(800, 10000, 2000)


@pytest.fixture
def base_params():
    """Returns a copy of the modified params_agile dictionary."""
    params = params_agile.copy()

    # Overwrite specific flags to ensure a controlled baseline for unit tests
    overrides = {
        "z": 2.0,
        "gflag": False,  # distinct galaxy test
        "lyForest": False,  # distinct forest test
        "bbnorm": 0.0,  # distinct blackbody test
        "ebv": 0.0,  # distinct reddening test
    }
    params.update(overrides)
    return params


# -----------------------------------------------------------------------------
# 1. Helper Function Tests
# -----------------------------------------------------------------------------


def test_luminosity_distance(cosmo):
    """Test luminosity distance calculation returns positive floats."""
    z = 2.0
    val = four_pi_dL_sq(z, cosmo=cosmo)
    assert isinstance(val, float)
    assert val > 0


def test_power_law():
    """Test the basic power law function logic."""
    wav = np.array([1000.0, 2000.0])
    slope = -2.0
    const = 10.0
    # Expected: 10 * wav^(-2) -> 10/1e6, 10/4e6
    res = pl(wav, slope, const)
    assert np.allclose(res, const * wav**slope)


def test_blackbody_shape():
    """Test blackbody returns Wien's law behavior (hotter = bluer).

    Note: The bb() function returns flux density per unit FREQUENCY (f_nu).
    The peak of f_nu occurs at lambda ~ 5100 / T (micron).
    The peak of f_lambda occurs at lambda ~ 2900 / T (micron).

    We convert to f_lambda for this test to match standard intuition.
    f_lambda = f_nu * (c / lambda^2) -> proportional to f_nu / lambda^2
    """
    wav = np.linspace(1000, 10000, 1000)
    temp = 5000  # Kelvin
    f_nu = bb(temp, wav)

    # Convert f_nu to f_lambda for peak checking
    # f_lambda propto f_nu / lambda^2
    f_lambda = f_nu / (wav**2)

    assert np.all(f_nu > 0), "Blackbody flux must be positive"

    # Peak for 5000K in f_lambda should be approx 5800 Angstroms
    peak_idx = np.argmax(f_lambda)
    peak_wav = wav[peak_idx]

    # Check peak is roughly in the visible range (4000-7000A)
    # 5800A falls comfortably in this range
    assert 4000 < peak_wav < 7000


def test_tau_eff():
    """Test Lyman alpha optical depth increases with redshift."""
    assert tau_eff(0.1) >= 0
    # Optical depth at z=6 should be significantly higher than z=2
    assert tau_eff(6.0) > tau_eff(2.0)


# -----------------------------------------------------------------------------
# 2. Main Class Logic Tests
# -----------------------------------------------------------------------------


def test_initialization_monotonicity(base_params, cosmo):
    """Test that non-monotonic wavelength arrays raise an exception."""
    bad_wav = np.array([1000, 1200, 1100])  # Not sorted
    with pytest.raises(Exception) as excinfo:
        Quasar_sed(wavlen=bad_wav, params=base_params, cosmo=cosmo)
    assert "monotonic" in str(excinfo.value)


def test_continuum_generation(base_params, mock_wavelengths, cosmo):
    """Test that a basic power-law continuum is generated."""
    # Isolate continuum by turning off lines, galaxy, dust, forest
    base_params["scal_emline"] = 0

    qs = Quasar_sed(wavlen=mock_wavelengths, params=base_params, cosmo=cosmo)
    # Check flux is generated and positive
    assert not np.all(qs.flux == 0)
    assert np.all(qs.flux >= 0)
    assert len(qs.flux) == len(mock_wavelengths)


def test_blackbody_addition(base_params, mock_wavelengths, cosmo):
    """Test that enabling blackbody (hot dust) adds an IR excess to the SED.

    We check that the Red/Blue flux ratio increases.
    """
    # 1. Baseline: No Blackbody
    base_params["bbnorm"] = 0.0
    qs_no_bb = Quasar_sed(wavlen=mock_wavelengths, params=base_params, cosmo=cosmo)

    # 2. With Blackbody (Hot dust, T~1200K, peaks in IR)
    base_params["bbnorm"] = 5.0
    base_params["tbb"] = 1200.0
    qs_with_bb = Quasar_sed(wavlen=mock_wavelengths, params=base_params, cosmo=cosmo)

    # Define "Blue" (UV) and "Red" (NIR) points
    # 2000A is far from the Blackbody peak; 8000A is closer
    idx_blue = (np.abs(mock_wavelengths - 2000)).argmin()
    idx_red = (np.abs(mock_wavelengths - 8000)).argmin()

    # Calculate Red-to-Blue ratios
    ratio_no_bb = qs_no_bb.flux[idx_red] / qs_no_bb.flux[idx_blue]
    ratio_with_bb = qs_with_bb.flux[idx_red] / qs_with_bb.flux[idx_blue]

    # The SED with the Blackbody should be "redder" (higher IR/UV ratio)
    assert ratio_with_bb > ratio_no_bb


def test_host_galaxy_integration(base_params, cosmo):
    """Test that host galaxy flux is added correctly."""
    # We must enable the galaxy flag
    base_params["gflag"] = True
    base_params["fragal"] = 0.5  # 50% galaxy contribution
    base_params["gplind"] = 0.5  # Ensure scaling isn't zero

    # Wavelength array MUST cover 4000-5000A for normalization check
    wav = np.linspace(3500, 5500, 1000)

    qs = Quasar_sed(wavlen=wav, params=base_params, cosmo=cosmo)

    # 1. Check host galaxy component is populated
    assert np.sum(qs.host_galaxy_flux) > 0

    # 2. Compare against pure quasar
    base_params["gflag"] = False
    qs_pure = Quasar_sed(wavlen=wav, params=base_params, cosmo=cosmo)

    # The composite flux should be higher than the pure AGN flux
    assert np.sum(qs.flux) > np.sum(qs_pure.flux)


def test_reddening(base_params, mock_wavelengths, cosmo):
    """Test that applying extinction (EBV > 0) reduces flux."""
    base_params["gflag"] = False

    # Case A: Unreddened
    qs_clean = Quasar_sed(
        wavlen=mock_wavelengths, params=base_params, ebv=0.0, cosmo=cosmo
    )

    # Case B: Reddened
    qs_dusty = Quasar_sed(
        wavlen=mock_wavelengths, params=base_params, ebv=0.1, cosmo=cosmo
    )

    # Dusty flux should be lower
    assert np.sum(qs_dusty.flux) < np.sum(qs_clean.flux)

    # Sanity check: Reddening usually affects blue light more than red.
    # Let's check ratio at 2000A vs 8000A
    idx_blue = (np.abs(mock_wavelengths - 2000)).argmin()
    idx_red = (np.abs(mock_wavelengths - 8000)).argmin()

    ratio_clean = qs_clean.flux[idx_blue] / qs_clean.flux[idx_red]
    ratio_dusty = qs_dusty.flux[idx_blue] / qs_dusty.flux[idx_red]

    # The dusty spectrum should be "redder", i.e., smaller Blue/Red ratio
    assert ratio_dusty < ratio_clean


def test_lyman_forest_suppression(base_params, cosmo):
    """Test that Lyman forest suppresses UV flux at high redshift."""
    z = 3.5
    # Wavelengths must cover rest-frame UV (800 - 1500 A)
    wav = np.linspace(800, 1500, 500)

    base_params["lyForest"] = True
    base_params["gflag"] = False

    qs = Quasar_sed(z=z, wavlen=wav, params=base_params, cosmo=cosmo)

    # 1. Hard Cutoff: Flux < lylim (912A) should be exactly zero
    limit_mask = wav < base_params["lylim"]
    assert np.all(qs.flux[limit_mask] == 0.0)

    # 2. Forest Region: 912 < lambda < 1216
    forest_mask = (wav > base_params["lylim"]) & (wav < 1216)

    # Create comparison model without forest
    base_params["lyForest"] = False
    qs_clear = Quasar_sed(z=z, wavlen=wav, params=base_params, cosmo=cosmo)

    # The forest should suppress flux in this region
    flux_forest = np.mean(qs.flux[forest_mask])
    flux_clear = np.mean(qs_clear.flux[forest_mask])

    assert flux_forest < flux_clear


def test_normalization_L3000(base_params, cosmo):
    """Test the flux normalization at 3000A."""
    wav = np.linspace(2500, 3500, 500)
    target_logL = 46.0
    z = 2.0

    qs = Quasar_sed(
        z=z, LogL3000=target_logL, wavlen=wav, params=base_params, cosmo=cosmo
    )

    # Calculate expected f_lambda at 3000A rest frame manually
    dL_term = four_pi_dL_sq(z, cosmo=cosmo)
    expected_f3000 = 10 ** (target_logL - dL_term) / (3000 * (1 + z))

    # Check class attribute matches calculation
    assert np.isclose(qs.f3000, expected_f3000)

    # Check actual flux array at 3000A
    idx_3000 = (np.abs(wav - 3000)).argmin()

    # Use loose tolerance because emission lines are added ON TOP of the continuum
    # which might shift the exact value at 3000A slightly
    assert np.isclose(qs.flux[idx_3000], expected_f3000, rtol=0.2)

    # Test with a quasar spectrum without emission lines for a tighter check
    base_params["scal_emline"] = 0.0
    qs_no_emlines = Quasar_sed(
        z=z, LogL3000=target_logL, wavlen=wav, params=base_params, cosmo=cosmo
    )
    assert np.isclose(qs_no_emlines.flux[idx_3000], expected_f3000, rtol=0.0001)


def test_convert_fnu_flambda(base_params, mock_wavelengths, cosmo):
    """Test helper method that converts f_nu to f_lambda.

    f_lambda = f_nu * c / lambda^2
    """
    qs = Quasar_sed(wavlen=mock_wavelengths, params=base_params, cosmo=cosmo)

    # Mock a flat f_nu (flux density per unit frequency)
    qs.flux = np.ones_like(mock_wavelengths)

    # Run conversion
    # Note: convert_fnu_flambda also normalizes, so we must account for that
    qs.convert_fnu_flambda(flxnrm=1.0, wavnrm=5000)

    # Since f_nu was constant, f_lambda should go as 1/lambda^2
    # pick two points
    lam1 = 2000.0
    lam2 = 4000.0

    val1 = np.interp(lam1, qs.wavlen, qs.flux)
    val2 = np.interp(lam2, qs.wavlen, qs.flux)

    # Ratio expected: (1/2000^2) / (1/4000^2) = 4000^2 / 2000^2 = 4
    ratio = val1 / val2
    assert np.isclose(ratio, 4.0, rtol=0.01)


def test_emission_lines_scaling(base_params, cosmo):
    """Test that changing the emission line scaling factor (scal_emline)
    actually changes the flux at a major line like Lyman Alpha (1216A) ."""
    # Use a dense grid near Lya to catch the peak
    wav = np.linspace(1150, 1300, 500)

    # 1. No lines
    base_params["scal_emline"] = 0.0
    qs_none = Quasar_sed(wavlen=wav, params=base_params, cosmo=cosmo)

    # 2. Positive scaling (scales by line flux)
    base_params["scal_emline"] = 1.0
    qs_strong = Quasar_sed(wavlen=wav, params=base_params, cosmo=cosmo)

    # 3. Check Flux at LyAlpha (approx 1216A)
    # The 'strong' model should have significantly higher flux at the line peak
    idx_lya = (np.abs(wav - 1216)).argmin()

    flux_none = qs_none.flux[idx_lya]
    flux_strong = qs_strong.flux[idx_lya]

    assert flux_strong > flux_none * 1.1  # Expecting at least 10% boost from lines


def test_balmer_continuum(base_params, cosmo):
    """Test the Balmer Continuum generation.

    The Balmer break is at ~3646 Angstroms. Flux shortward of 3646A
    should be boosted when BC is enabled .
    """
    # Wavelengths around the Balmer break (3646A)
    wav = np.linspace(3000, 4000, 1000)

    # 1. Without Balmer Continuum
    base_params["bcnorm"] = 0.0
    qs_no_bc = Quasar_sed(wavlen=wav, params=base_params, cosmo=cosmo)

    # 2. With Balmer Continuum
    base_params["bcnorm"] = 1.0  # Enabled (flux density normalization)
    base_params["tbc"] = 15000.0  # Standard BC temp

    qs_bc = Quasar_sed(wavlen=wav, params=base_params, cosmo=cosmo)

    # Define Shortward (UV) and Longward (Optical) points
    # The BC adds flux primarily SHORTWARD of 3646A
    idx_short = (np.abs(wav - 3500)).argmin()  # < 3646
    idx_long = (np.abs(wav - 3800)).argmin()  # > 3646

    # Calculate the ratio of Flux(With BC) / Flux(No BC)
    ratio_short = qs_bc.flux[idx_short] / qs_no_bc.flux[idx_short]
    ratio_long = qs_bc.flux[idx_long] / qs_no_bc.flux[idx_long]

    # The boost shortward of the break should be stronger than longward
    # (Because BC is an emission feature that "turns on" below 3646A)
    assert ratio_short > ratio_long


# -----------------------------------------------------------------------------
# 3. Coverage Extension Tests
# -----------------------------------------------------------------------------


def test_default_params_and_unknown_kwargs(capsys, mock_wavelengths, cosmo):
    """
    Covers:
      - Default params loading (via default argument params=default_params)
      - 'Warning: "{}" not recognised as a kwarg'
    """
    # Initialize without 'params', forcing use of default params
    # Pass an unknown keyword argument 'garbage_param'
    qs = Quasar_sed(wavlen=mock_wavelengths, garbage_param=12345, cosmo=cosmo)

    # Check that params were loaded successfully (default isn't empty)
    assert qs.params is not None
    assert "garbage_param" in qs.params

    # Capture stdout to check for the warning print statement
    captured = capsys.readouterr()
    assert 'Warning: "garbage_param" not recognised as a kwarg' in captured.out


def test_no_logl3000(base_params, mock_wavelengths, cosmo):
    """
    Covers:
      - 'else: self.convert_fnu_flambda()' (when LogL3000 is None)
    """
    qs = Quasar_sed(
        z=2.0, LogL3000=None, wavlen=mock_wavelengths, params=base_params, cosmo=cosmo
    )
    # Ensure flux is still generated and positive (normalization happened)
    assert np.all(qs.flux >= 0)
    assert np.sum(qs.flux) > 0


def test_emline_type_defaults(base_params, mock_wavelengths, cosmo):
    """
    Covers:
      - 'else: self.emline_type = 0.0'
      - 'if varlin == 0.0:' (Average emission line template logic)
    """
    # Set conditions to hit the default emission line type logic
    base_params["emline_type"] = None
    base_params["beslope"] = 0.0  # Disable Baldwin effect

    qs = Quasar_sed(wavlen=mock_wavelengths, params=base_params, cosmo=cosmo)

    # Internally, emline_type should have been set to 0.0
    assert qs.emline_type == 0.0

    # Verify emission lines were added (check flux vs continuum only)
    base_params["scal_emline"] = 0
    qs_cont = Quasar_sed(wavlen=mock_wavelengths, params=base_params, cosmo=cosmo)

    # Integrated flux with lines (default type 0) should be > continuum
    assert np.sum(qs.flux) > np.sum(qs_cont.flux)


def test_host_galaxy_wavelength_error(base_params, cosmo):
    """
    Covers:
      - 'wavlen must cover 4000-5000 A for galaxy normalisation' exception
    """
    base_params["gflag"] = True

    # Wavelength array that does NOT cover 4000-5000A
    bad_wav = np.linspace(1000, 3000, 500)

    with pytest.raises(Exception) as excinfo:
        Quasar_sed(wavlen=bad_wav, params=base_params, cosmo=cosmo)

    assert "wavlen must cover 4000-5000 A" in str(excinfo.value)

    # Wavlength array that partially covers 4000-5000A
    bad_wav2 = np.linspace(4500, 6000, 500)

    with pytest.raises(Exception) as excinfo2:
        Quasar_sed(wavlen=bad_wav2, params=base_params, cosmo=cosmo)

    assert "wavlen must cover 4000-5000 A" in str(excinfo2.value)
