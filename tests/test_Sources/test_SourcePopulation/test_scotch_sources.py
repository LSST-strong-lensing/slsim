import h5py
import pytest
import numpy as np
import astropy.units as u
import slsim.Sources.SourcePopulation.scotch_sources as scotch_module

from pathlib import Path
from types import SimpleNamespace
from slsim.Sources.source import Source
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.SourceTypes.general_lightcurve import GeneralLightCurve
from slsim.Sources.SourceTypes.point_plus_extended_source import PointPlusExtendedSource

# -----------------------------
# Helpers / Fixtures, since we
# aren't using conftest.py.
# Remember to pester devs about
# this in the future
# -----------------------------


@pytest.fixture(scope="function")
def scotch_h5(tmp_path: Path):
    """
    Create a minimal SCOTCH-like HDF5 file with:
      - Transient classes: SNII (active), AGN (present but no survivors)
      - SNII has two subclasses:
          * A: 2 rows (1 passes cuts, 1 fails via faint mag_r)
          * B: 1 row (passes)
      - AGN has one subclass with a too-faint mag_r so it fails cuts
      - Host table for each class; SNII host[0] is valid, host[1] is 'hostless' (z=999)
    """
    path = tmp_path / "scotch_test.h5"
    with h5py.File(path, "w") as f:
        tt = f.create_group("TransientTable")
        ht = f.create_group("HostTable")

        # ---- SNII hosts ----
        sn_host = ht.create_group("SNII")
        gids_sn = np.array([b"00000001", b"00000002"])  # |S8
        sn_host.create_dataset("GID", data=gids_sn)
        sn_host.create_dataset("z", data=np.array([0.5, 999.0]))  # 999 => hostless
        sn_host.create_dataset("a_rot", data=np.array([45.0, 30.0]))  # degrees
        for name in [
            "a0",
            "b0",
            "a1",
            "b1",
            "ellipticity0",
            "ellipticity1",
            "n0",
            "n1",
        ]:
            sn_host.create_dataset(name, data=np.array([1.0, 1.5]))
        sn_host.create_dataset("w0", data=np.array([0.3, 0.4]))
        sn_host.create_dataset("w1", data=np.array([0.7, 0.6]))

        # Host mag for 'r' band so that first passes and second fails
        sn_host.create_dataset("mag_r", data=np.array([21.0, 25.0]))

        # ---- AGN hosts ---- (present but transients fail cuts; class will be inactive)
        agn_host = ht.create_group("AGN")
        gids_agn = np.array([b"10000001"])
        agn_host.create_dataset("GID", data=gids_agn)
        agn_host.create_dataset("z", data=np.array([0.7]))
        agn_host.create_dataset("a_rot", data=np.array([0.0]))
        for name in [
            "a0",
            "b0",
            "a1",
            "b1",
            "ellipticity0",
            "ellipticity1",
            "n0",
            "n1",
        ]:
            agn_host.create_dataset(name, data=np.array([1.0]))
        agn_host.create_dataset("w0", data=np.array([0.3, 0.4]))
        agn_host.create_dataset("w1", data=np.array([0.7, 0.6]))
        agn_host.create_dataset("mag_r", data=np.array([30.0]))  # too faint

        # ---- SNII transients ----
        sn_tt = tt.create_group("SNII")

        # Subclass A: 2 rows; first passes (mag_r <= 22), second fails (all 99s -> NaNs)
        A = sn_tt.create_group("SNII-Templates")
        A.create_dataset("z", data=np.array([0.6, 0.4]))
        A.create_dataset("GID", data=np.array([gids_sn[0], gids_sn[1]]))
        A.create_dataset("ra_off", data=np.array([0.01, -0.01]))
        A.create_dataset("dec_off", data=np.array([0.02, -0.02]))
        A.create_dataset("MJD", data=np.tile(np.array([1.0, 2.0, 3.0]), (2, 1)))
        for b in ("u", "g", "r", "i", "z", "Y"):
            if b == "r":
                A.create_dataset(
                    "mag_r", data=np.array([[23.0, 22.0, 23.0], [99.0, 99.0, 99.0]])
                )
            else:
                A.create_dataset(
                    f"mag_{b}", data=np.array([[99.0, 99.0, 99.0], [99.0, 99.0, 99.0]])
                )

        # Subclass B: 1 row; passes
        B = sn_tt.create_group("SNII+HostXT_V19")
        B.create_dataset("z", data=np.array([0.55]))
        B.create_dataset("GID", data=np.array([gids_sn[0]]))
        B.create_dataset("ra_off", data=np.array([0.0]))
        B.create_dataset("dec_off", data=np.array([0.0]))
        B.create_dataset("MJD", data=np.array([[10.0, 11.0, 12.0]]))
        for b in ("u", "g", "r", "i", "z", "Y"):
            if b == "r":
                B.create_dataset("mag_r", data=np.array([[20.0, 20.0, 20.0]]))
            else:
                B.create_dataset(f"mag_{b}", data=np.array([[99.0, 99.0, 99.0]]))

        # ---- AGN transients ---- (should fail)
        agn_tt = tt.create_group("AGN")
        X = agn_tt.create_group("AGN")
        X.create_dataset("z", data=np.array([0.8]))
        X.create_dataset("GID", data=np.array([gids_agn[0]]))
        X.create_dataset("ra_off", data=np.array([0.0]))
        X.create_dataset("dec_off", data=np.array([0.0]))
        X.create_dataset("MJD", data=np.array([[5.0, 6.0, 7.0]]))
        for b in ("u", "g", "r", "i", "z", "Y"):
            if b == "r":
                X.create_dataset(
                    "mag_r", data=np.array([[30.0, 30.0, 30.0]])
                )  # too faint
            else:
                X.create_dataset(f"mag_{b}", data=np.array([[99.0, 99.0, 99.0]]))

    return path


@pytest.fixture
def scotch_instance(scotch_h5):
    """Construct a ScotchSources with deterministic RNG and an r-band cut
    (<=22)."""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # sky_area = 1.0 * u.deg**2
    kwargs_cut = {"band": ["r"], "band_max": [22.0]}
    inst = scotch_module.ScotchSources(
        cosmo=cosmo,
        sky_area=None,
        scotch_path=scotch_h5,
        transient_types=None,
        kwargs_cut=kwargs_cut,
        rng=np.random.default_rng(123),
    )
    return inst


@pytest.fixture(scope="function")
def scotch_h5_unknown_subclass(tmp_path: Path):
    """Minimal file with a subclass not in RATE_FUNCS and not AGN; should raise
    in _get_expected_number."""
    p = tmp_path / "scotch_unknown_subclass.h5"
    with h5py.File(p, "w") as f:
        tt = f.create_group("TransientTable")
        ht = f.create_group("HostTable")

        # Host table for SNII
        sn_host = ht.create_group("SNII")
        sn_host.create_dataset("GID", data=np.array([b"12345678"]))
        sn_host.create_dataset("z", data=np.array([0.2]))
        sn_host.create_dataset("a_rot", data=np.array([0.0]))
        for name in [
            "a0",
            "b0",
            "a1",
            "b1",
            "ellipticity0",
            "ellipticity1",
            "n0",
            "n1",
        ]:
            sn_host.create_dataset(name, data=np.array([1.0]))
        sn_host.create_dataset("w0", data=np.array([0.5]))
        sn_host.create_dataset("w1", data=np.array([0.5]))
        sn_host.create_dataset("mag_r", data=np.array([20.0]))

        # Transient subclass with unknown name that survives
        sn_tt = tt.create_group("SNII")
        sub = sn_tt.create_group("SNII-NotInRates")
        sub.create_dataset("z", data=np.array([0.3]))
        sub.create_dataset("GID", data=np.array([b"12345678"]))
        sub.create_dataset("ra_off", data=np.array([0.0]))
        sub.create_dataset("dec_off", data=np.array([0.0]))
        sub.create_dataset("MJD", data=np.array([[1.0, 2.0, 3.0]]))
        for b in ("u", "g", "r", "i", "z", "Y"):
            vals = [20.0, 20.0, 20.0] if b == "r" else [99.0, 99.0, 99.0]
            sub.create_dataset(f"mag_{b}", data=np.array([vals]))
    return p


@pytest.fixture(scope="function")
def scotch_twofile_missing_host(scotch_h5, tmp_path: Path):
    """Second file that has TransientTable/SNII but NO HostTable/SNII.

    Exercises 'no_host_table' and the per-file host-info stubs.
    """
    p2 = tmp_path / "scotch_missing_host.h5"
    with h5py.File(p2, "w") as f:
        tt = f.create_group("TransientTable")
        tt_sn = tt.create_group("SNII")
        sub = tt_sn.create_group("SNII-Templates")
        sub.create_dataset("z", data=np.array([0.3]))
        sub.create_dataset("GID", data=np.array([b"00000001"]))
        sub.create_dataset("ra_off", data=np.array([0.0]))
        sub.create_dataset("dec_off", data=np.array([0.0]))
        sub.create_dataset("MJD", data=np.array([[1.0, 2.0, 3.0]]))
        for b in ("u", "g", "r", "i", "z", "Y"):
            vals = [20.0, 20.0, 20.0] if b == "r" else [99.0, 99.0, 99.0]
            sub.create_dataset(f"mag_{b}", data=np.array([vals]))
        # Create HostTable group but leave it EMPTY for SNII (no 'SNII' key)
        f.create_group("HostTable")
    return [scotch_h5, p2]


@pytest.fixture(scope="function")
def scotch_twofile_with_hostless_survivor(scotch_h5, tmp_path: Path):
    """Second file adds a hostless (z==999) SNII row that PASSES cuts.

    This lets us exercise has_host=False all the way through
    _draw_source_dict.
    """
    p2 = tmp_path / "scotch_hostless_ok.h5"
    with h5py.File(p2, "w") as f:
        tt = f.create_group("TransientTable")
        ht = f.create_group("HostTable")

        # Host table for SNII with a hostless entry that also passes mag cuts
        sn_host = ht.create_group("SNII")
        sn_host.create_dataset("GID", data=np.array([b"00000003"]))
        sn_host.create_dataset("z", data=np.array([999.0]))  # hostless
        sn_host.create_dataset("a_rot", data=np.array([0.0]))
        for name in [
            "a0",
            "b0",
            "a1",
            "b1",
            "ellipticity0",
            "ellipticity1",
            "n0",
            "n1",
        ]:
            sn_host.create_dataset(name, data=np.array([1.0]))
        sn_host.create_dataset("w0", data=np.array([0.5]))
        sn_host.create_dataset("w1", data=np.array([0.5]))
        sn_host.create_dataset(
            "mag_r", data=np.array([20.0])
        )  # bright to pass host mag cuts

        # Transient subclass referencing the hostless GID; bright LC so it survives
        sn_tt = tt.create_group("SNII")
        sub = sn_tt.create_group("SNII-Templates")
        sub.create_dataset("z", data=np.array([0.3]))
        sub.create_dataset("GID", data=np.array([b"00000003"]))
        sub.create_dataset("ra_off", data=np.array([0.0]))
        sub.create_dataset("dec_off", data=np.array([0.0]))
        sub.create_dataset("MJD", data=np.array([[1.0, 2.0, 3.0]]))
        for b in ("u", "g", "r", "i", "z", "Y"):
            vals = [20.0, 20.0, 20.0] if b == "r" else [99.0, 99.0, 99.0]
            sub.create_dataset(f"mag_{b}", data=np.array([vals]))
    return [scotch_h5, p2]


# ----- reference (oracle) formulas (duplicated on purpose for independence) -----
def d08_ref(z):  # Dilday+08
    return (1 + z) ** 1.5


def md14_ref(z):  # Madau & Dickinson 2014, Eq. 15
    return (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)


def s15_ref(z):  # Strolger+15, Eq. 9
    return (1 + z) ** 5.0 / (1 + ((1 + z) / 1.5) ** 6.1)


def snia_rate_ref(z):
    r0 = 25.0
    z = np.asarray(z)
    return np.where(z < 1, r0 * d08_ref(z), r0 * (1 + z) ** -0.5)


def snia_91bg_rate_ref(z):
    return 3.0 * d08_ref(z)


def sniax_rate_ref(z):
    return 6.0 * md14_ref(z)


def snii_rate_ref(z):
    return 45.0 * s15_ref(z)


def snibc_rate_ref(z):
    return 19.0 * s15_ref(z)


def slsn_rate_ref(z):
    return 0.02 * md14_ref(z)


def tde_rate_ref(z):
    r0 = 1.0
    z = np.asarray(z)
    return r0 * 10 ** (-5 * z / 6)


def kn_rate_ref(z):
    z = np.asarray(z)
    return 6.0 * np.ones_like(z)


# Map functions-under-test to their oracle
CASES = [
    (scotch_module.d08, d08_ref),
    (scotch_module.md14, md14_ref),
    (scotch_module.s15, s15_ref),
    (scotch_module.snia_rate, snia_rate_ref),
    (scotch_module.snia_91bg_rate, snia_91bg_rate_ref),
    (scotch_module.sniax_rate, sniax_rate_ref),
    (scotch_module.snii_rate, snii_rate_ref),
    (scotch_module.snibc_rate, snibc_rate_ref),
    (scotch_module.slsn_rate, slsn_rate_ref),
    (scotch_module.tde_rate, tde_rate_ref),
    (scotch_module.kn_rate, kn_rate_ref),
]

# Scalars and arrays to hit both code paths & vectorization
ZS = [
    0.0,
    0.3,
    0.999999,  # just below boundary for snia_rate
    1.0,  # boundary
    2.1,
    np.array([0.0, 0.2, 0.9999, 1.0, 3.5]),
    np.linspace(0, 5, 11),
]


# --- Tiny, dependency-free cosmology doubles ---
class FakeCosmoConst:
    """D(dV)/dz/dΩ = C (constant)"""

    def __init__(self, C: float):
        self.C = C

    def differential_comoving_volume(self, z):
        # expected_number reads .value
        return SimpleNamespace(value=self.C)


class FakeCosmoLinear:
    """D(dV)/dz/dΩ = a*z + b (linear in z)"""

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b

    def differential_comoving_volume(self, z):
        return SimpleNamespace(value=self.a * z + self.b)


A, B, R = 3.0, 5.0, 12


def constant_rate_fn(z):
    return R


def linear_rate_fn(z):
    return A * z + B


# -----------------------------
# Actual tests
# -----------------------------


@pytest.mark.parametrize("fn,ref", CASES)
@pytest.mark.parametrize("z", ZS)
def test_rate_formulas_match_reference(fn, ref, z):
    got = fn(z)
    exp = ref(z)
    # Uniform comparison for scalars and arrays
    assert np.allclose(got, exp, rtol=1e-12, atol=0.0)


def test_snia_rate_piecewise_boundary_behavior():
    # Explicitly assert branch selection around z=1
    z_below = np.array([0.0, 0.5, 0.999999])
    z_edge = 1.0
    z_above = np.array([1.000001, 2.0, 3.0])

    r0 = 25.0
    # Left side (<1) uses d08
    left = scotch_module.snia_rate(z_below)
    left_exp = r0 * d08_ref(z_below)
    assert np.allclose(left, left_exp, rtol=1e-12)

    # Exactly at 1.0 uses the >=1 branch per the implementation (z < 1)
    at_edge = scotch_module.snia_rate(z_edge)
    at_edge_exp = r0 * (1 + z_edge) ** -0.5
    assert np.allclose(at_edge, at_edge_exp, rtol=1e-12)

    # Right side (>=1) uses power-law decline
    right = scotch_module.snia_rate(z_above)
    right_exp = r0 * (1 + z_above) ** -0.5
    assert np.allclose(right, right_exp, rtol=1e-12)


def test_rate_shapes_and_types_are_preserved():
    # Arrays in -> arrays out with same shape
    z = np.random.RandomState(0).rand(7, 3) * 5.0
    for fn, _ in CASES:
        out = fn(z)
        assert isinstance(out, np.ndarray)
        assert out.shape == z.shape

    # Scalars in -> numeric scalar (numpy scalar or Python float both okay)
    z_scalar = 0.7
    for fn, _ in CASES:
        out = fn(z_scalar)
        assert np.isscalar(out) or (isinstance(out, np.ndarray) and out.shape == ())


def test_basic_rate_invariants():
    # Non-negativity for z >= 0 for all these models
    z = np.linspace(0, 10, 101)
    for fn, _ in CASES:
        out = fn(z)
        assert np.all(out >= 0)

    # kn_rate is constant 6 for any z
    z2 = np.array([0.0, 1.0, 5.0, 10.0])
    assert np.allclose(scotch_module.kn_rate(z2), 6.0)


def test_expected_number_constant_rate_and_volume():
    """If rate_fn(z) = R (constant) and dV/dz/dΩ = C (constant),

    integrand = 4π * (C) * 1e-6 * (R), so
    N = 4π * C * 1e-6 * R * (z_max - z_min)
    """

    C = 3.0
    z0, z1 = 0.2, 2.7

    cosmo = FakeCosmoConst(C)
    got = scotch_module.expected_number(constant_rate_fn, cosmo, z0, z1)
    exp = 4 * np.pi * C * 1e-6 * R * (z1 - z0)

    assert np.isclose(got, exp, rtol=1e-12, atol=0.0)


def test_expected_number_linear_rate_linear_volume_closed_form():
    """For rate_fn(z) = A*z + B and dV/dz/dΩ = a*z + b,

    integrand = 4π * 1e-6 * (a z + b)(A z + B)           = 4π * 1e-6 *
    [aA z^2 + (aB + bA) z + bB] Integrate term-wise on [z0, z1].
    """
    a, b = 2.0, 1.0  # volume coefficients
    z0, z1 = 0.4, 1.9

    cosmo = FakeCosmoLinear(a, b)

    got = scotch_module.expected_number(linear_rate_fn, cosmo, z0, z1)

    k = 4 * np.pi * 1e-6
    exp = k * (
        (a * A) / 3.0 * (z1**3 - z0**3)
        + (a * B + b * A) / 2.0 * (z1**2 - z0**2)
        + (b * B) * (z1 - z0)
    )

    assert np.isclose(got, exp, rtol=1e-12, atol=0.0)


def test_expected_number_zero_when_same_limits():
    cosmo = FakeCosmoConst(10.0)
    z = 1.2345
    got = scotch_module.expected_number(constant_rate_fn, cosmo, z, z)
    assert np.isclose(got, 0.0, atol=0.0, rtol=0.0)


def test_expected_number_defaults_integrate_0_to_3():
    """With no z_min/z_max passed, integrate over [0, 3].

    Use constant rate & constant volume for a closed-form check.
    """

    C = 4.0
    cosmo = FakeCosmoConst(C)
    got = scotch_module.expected_number(constant_rate_fn, cosmo)  # use defaults
    exp = 4 * np.pi * C * 1e-6 * R * (3.0 - 0.0)

    assert np.isclose(got, exp, rtol=1e-12, atol=0.0)


def test_expected_number_kwargs_override_defaults_individually():
    """If only z_max is provided, z_min should remain at 0.0 (default)."""
    C = 2.0
    cosmo = FakeCosmoConst(C)

    got = scotch_module.expected_number(constant_rate_fn, cosmo, z_max=5.0)
    exp = 4 * np.pi * C * 1e-6 * R * (5.0 - 0.0)

    assert np.isclose(got, exp, rtol=1e-12, atol=0.0)


def test_norm_band_names():
    _norm = scotch_module._norm_band_names
    assert _norm(["U", "g", "Y", " y  "]) == ["u", "g", "Y", "Y"]


def test_galaxy_projected_eccentricity():
    e1, e2 = scotch_module.galaxy_projected_eccentricity(
        ellipticity=0.0, rotation_angle=None
    )
    assert np.isclose(e1, 0.0) and np.isclose(e2, 0.0)


def test_galaxy_projected_eccentricity_with_explicit_angle():
    # Covers explicit-angle branch: e1=e*cos(2phi), e2=e*sin(2phi)
    eps = 0.5
    phi = np.pi / 6  # 30 deg
    e = scotch_module.param_util.epsilon2e(eps)
    e1, e2 = scotch_module.galaxy_projected_eccentricity(eps, rotation_angle=phi)
    assert np.isclose(e1, e * np.cos(2 * phi))
    assert np.isclose(e2, e * np.sin(2 * phi))


def test_init_exclude_agn_flag(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    inst = scotch_module.ScotchSources(
        cosmo=cosmo,
        scotch_path=scotch_h5,
        rng=0,
        sample_uniformly=True,  # ensure AGN would otherwise be included
        exclude_agn=True,
    )
    assert "AGN" not in inst.transient_types
    assert "AGN" not in inst.active_transient_types


def test_init_warning_no_objects_passed(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    kwargs_cut = {"band": ["r"], "band_max": [22.0]}
    match_string = (
        "has no objects passing the provided "
        + "kwargs_cut filters and will be ignored"
    )
    with pytest.warns(UserWarning, match=match_string):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            sky_area=sky_area,
            scotch_path=scotch_h5,
            transient_types=None,
            kwargs_cut=kwargs_cut,
            rng=np.random.default_rng(123),
        )


def test_init_unknown_transient_type_raises(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            sky_area=sky_area,
            scotch_path=scotch_h5,
            transient_types=["DOES_NOT_EXIST"],
        )


def test_init_unknown_transient_subtype_raises(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            sky_area=sky_area,
            scotch_path=scotch_h5,
            transient_subtypes={"SNII": "DOES_NOT_EXIST"},
        )


def test_unknown_subclass_raises_expected_number(scotch_h5_unknown_subclass):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    with pytest.raises(KeyError, match="not found in rate functions"):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            scotch_path=scotch_h5_unknown_subclass,
            kwargs_cut={"band": "r", "band_max": 22.0},
        )


def test_init_kwargs_cut_requires_matching_keys(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # band without band_max
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo, scotch_path=scotch_h5, kwargs_cut={"band": ["r"]}
        )
    # band_max without band
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo, scotch_path=scotch_h5, kwargs_cut={"band_max": [22.0]}
        )


def test_init_invalid_band_spec_raises(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            sky_area=sky_area,
            scotch_path=scotch_h5,
            kwargs_cut={"band": ["r", "g"], "band_max": [22.0]},
        )


def test_init_band_cuts_as_str(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    scotch = scotch_module.ScotchSources(
        cosmo=cosmo,
        sky_area=sky_area,
        scotch_path=scotch_h5,
        kwargs_cut={"band": "r", "band_max": 22.0},
    )

    assert isinstance(scotch.bands_to_filter, list)
    assert isinstance(scotch.band_maxes, list)
    assert len(scotch.bands_to_filter) == 1
    assert len(scotch.band_maxes) == 1
    assert "r" in scotch.bands_to_filter
    assert 22.0 in scotch.band_maxes


def test_init_unsupported_band_raises(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            sky_area=sky_area,
            scotch_path=scotch_h5,
            kwargs_cut={"band": ["q"], "band_max": [22.0]},
        )


def test_init_uniform_sampling(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    scotch = scotch_module.ScotchSources(
        cosmo=cosmo, scotch_path=scotch_h5, sample_uniformly=True
    )

    class_weights = scotch.class_weights
    print(scotch.active_transient_types)
    print(scotch.transient_subtypes)
    assert np.all(class_weights == np.array([1, 2]) / 3)
    assert np.sum(class_weights) == 1.0

    snii_subclass_weights = scotch._index["SNII"].subclass_weights
    assert np.all(snii_subclass_weights == 0.5)

    agn_subclass_weights = scotch._index["AGN"].subclass_weights
    assert np.all(agn_subclass_weights == 1.0)


def test_sky_area_scaling_with_stubbed_expected_number(scotch_h5, monkeypatch):
    # Make expected_number tiny and deterministic so we can assert scaling
    def _stub_expected_number(rate_fn, cosmo, z_min=0.0, z_max=3.0):
        return 2.0  # per subclass

    # AGN are just set to total_ok = 1 in this case

    monkeypatch.setattr(scotch_module, "expected_number", _stub_expected_number)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # No cuts -> active subclasses: SNII-Templates, SNII+HostXT_V19, AGN
    base = scotch_module.ScotchSources(cosmo=cosmo, scotch_path=scotch_h5, rng=0)
    # total_expected = 2 * 2 + 1= 5; n_source_selected should be 4 with this file
    print(base.active_transient_types)
    print(base.transient_subtypes)
    print(base.total_expected)
    print(base.n_source_selected)
    print(base._effective_sky_area)
    print(scotch_module.SKY_AREA)

    assert base.total_expected == 5
    assert base.source_number_selected == 4
    assert base._effective_sky_area == scotch_module.SKY_AREA * 4 / 5

    # effective_sky_area = SKY_AREA * 4 / 5
    # If we pass sky_area = 4 * effective => scaling factor = 4,
    # new n_source_selected = 4 * 4 = 16.
    sky_area = (4 * (scotch_module.SKY_AREA * 4 / 5)) * u.deg**2
    scaled = scotch_module.ScotchSources(
        cosmo=cosmo, scotch_path=scotch_h5, rng=0, sky_area=sky_area
    )
    assert scaled.source_number_selected == 16


def test_missing_host_table_is_skipped(scotch_twofile_missing_host):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    inst = scotch_module.ScotchSources(
        cosmo=cosmo,
        scotch_path=scotch_twofile_missing_host,
        kwargs_cut={"band": "r", "band_max": 22.0},
        rng=0,
    )
    # Totals should match the single-file SNII (3 rows total, 2 survivors),
    # i.e., the second file's SNII shard is skipped due to missing host table.
    ci = inst._index["SNII"]
    assert ci.total == 3
    assert ci.total_selected == 2


def test_no_objects_pass_cut(scotch_h5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    with pytest.raises(ValueError):
        scotch_module.ScotchSources(
            cosmo=cosmo,
            sky_area=sky_area,
            scotch_path=scotch_h5,
            kwargs_cut={"band": "r", "band_max": -np.inf},
        )


def test_host_pass_mask(scotch_instance):
    host_grp = scotch_instance._index["SNII"].host_grp[0]
    mask = scotch_instance._host_pass_mask(host_grp)
    assert mask.dtype == bool
    assert mask.shape == host_grp["z"].shape
    assert mask.tolist() == [True, False]


def test_transient_pass_mask_and_selection(scotch_instance):
    cls = "SNII"
    ci = scotch_instance._index[cls]
    # Subclass "SNII-Templates" -> [True, False]
    subA = next(s for s in ci.subclasses if s.name == "SNII-Templates")
    subA = subA.shards[0]
    maskA = scotch_instance._transient_pass_mask(
        subA.grp, ci.host_gid_sorted[0], ci.host_mask_sorted[0], batch=1
    )
    assert maskA.tolist() == [True, False]
    # Subclass "SNII+HostXT_V19" -> [True]
    subB = next(s for s in ci.subclasses if s.name == "SNII+HostXT_V19")
    subB = subB.shards[0]
    maskB = scotch_instance._transient_pass_mask(
        subB.grp, ci.host_gid_sorted[0], ci.host_mask_sorted[0], batch=1
    )
    assert maskB.tolist() == [True]
    # Totals reflect only active classes with survivors
    assert scotch_instance.source_number == 3  # total rows in SNII (2 + 1)
    assert scotch_instance.source_number_selected == 2  # two survivors


def test_sample_from_class_yields_valid_indices(scotch_instance):
    s, sh, i = scotch_instance._sample_from_class("SNII")
    # With our data: both subclasses only have index 0 eligible

    assert i == 0
    assert sh.N >= 1
    assert sh.n_ok >= 1


def test_host_lookup(scotch_instance):
    gid = scotch_instance._index["SNII"].host_grp[0]["GID"][0]
    print(gid)
    idx = scotch_instance._host_lookup("SNII", 0, gid)
    assert idx == 0
    with pytest.raises(KeyError):
        scotch_instance._host_lookup("SNII", 0, b"99999999")


def test_build_host_dict_and_hostless(scotch_instance):
    host_grp = scotch_instance._index["SNII"].host_grp[0]
    d0 = scotch_instance._build_host_dict(host_grp, 0)
    # Basic keys + converted names + computed ones
    for k in [
        "ellipticity0",
        "ellipticity1",
        "a_rot",
        "a0",
        "b0",
        "a1",
        "b1",
        "n_sersic_0",
        "n_sersic_1",
        "e0_1",
        "e0_2",
        "e1_1",
        "e1_2",
        "angular_size_0",
        "angular_size_1",
        "w0",
        "w1",
    ]:
        assert k in d0
    # a_rot converted to radians
    assert np.isclose(d0["a_rot"], np.deg2rad(45.0))
    # Host 1 is hostless (z==999.0)
    d1 = scotch_instance._build_host_dict(host_grp, 1)
    assert d1 == {}


def test_draw_source_dict(scotch_instance):
    source_dict, has_host = scotch_instance._draw_source_dict()
    # Transient metadata present
    for k in ("name", "z", "ra_off", "dec_off"):
        assert k in source_dict

    # Host metadata present if has_host
    if has_host:
        for k in [
            "ellipticity0",
            "ellipticity1",
            "a_rot",
            "a0",
            "b0",
            "a1",
            "b1",
            "n_sersic_0",
            "n_sersic_1",
            "e0_1",
            "e0_2",
            "e1_1",
            "e1_2",
            "angular_size_0",
            "angular_size_1",
            "w0",
            "w1",
        ]:
            assert k in source_dict
    else:
        for k in [
            "ellipticity0",
            "ellipticity1",
            "a_rot",
            "a0",
            "b0",
            "a1",
            "b1",
            "n_sersic_0",
            "n_sersic_1",
            "e0_1",
            "e0_2",
            "e1_1",
            "e1_2",
            "angular_size_0",
            "angular_size_1",
            "w0",
            "w1",
        ]:
            assert k not in source_dict

    # Lightcurve keys
    assert "MJD" in source_dict and source_dict["MJD"].ndim == 1
    for b in ("u", "g", "r", "i", "z", "Y"):
        assert f"ps_mag_{b}" in source_dict
        assert source_dict[f"ps_mag_{b}"].ndim == 1


def test_draw_source_dict_hostless_and_mjd_zero(
    scotch_twofile_with_hostless_survivor, monkeypatch
):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    inst = scotch_module.ScotchSources(
        cosmo=cosmo,
        scotch_path=[str(p) for p in scotch_twofile_with_hostless_survivor],
        kwargs_cut={"band": "r", "band_max": 22.0},
        rng=0,
    )

    # Find the shard/row that references the hostless GID b"00000003"
    target_cls = "SNII"
    target_gid = b"00000003"
    ci = inst._index[target_cls]
    chosen = None
    for s in ci.subclasses:
        for sh in s.shards:
            gids = sh.grp["GID"][:]
            hits = np.where(gids == target_gid)[0]
            if hits.size:
                i_file = int(hits[0])
                # Ensure this row is eligible (either all-ok int or contained in array)
                if isinstance(sh.eligible, (int, np.integer)):
                    eligible_index = i_file
                else:
                    if i_file not in set(map(int, sh.eligible.tolist())):
                        continue
                    eligible_index = i_file
                chosen = (s, sh, eligible_index)
                break
        if chosen:
            break
    assert chosen is not None, "Could not locate the hostless surviving row"

    # Force _draw_source_dict to use the hostless row
    monkeypatch.setattr(inst, "_sample_from_class", lambda cls: chosen)
    source_dict, has_host = inst._draw_source_dict()
    assert has_host is False
    # MJD should be zeroed at peak => contains 0
    assert 0.0 in source_dict["MJD"]

    # draw_source should now produce a pure GeneralLightCurve (no extended host)
    src = inst.draw_source()
    assert isinstance(src._source, GeneralLightCurve)


def test_draw_source(scotch_instance):

    src = scotch_instance.draw_source()
    assert isinstance(src, Source)
    assert isinstance(src._source, GeneralLightCurve) or isinstance(
        src._source, PointPlusExtendedSource
    )


def test_close(scotch_instance):
    scotch_instance.close()
    for file in scotch_instance.files:
        assert hasattr(file, "id")
        assert not file.id.valid
