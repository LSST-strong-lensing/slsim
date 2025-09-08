import h5py
import pytest
import numpy as np
import astropy.units as u
import slsim.Sources.SourcePopulation.scotch_sources as scotch_module

from pathlib import Path
from astropy.cosmology import FlatLambdaCDM

# -----------------------------
# Helpers / Fixtures, since we aren't using conftest.py
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
        for name in ["a0", "b0", "a1", "b1", "ellipticity0", "ellipticity1", "n0", "n1"]:
            sn_host.create_dataset(name, data=np.array([1.0, 1.5]))
        # Host mag for 'r' band so that first passes and second fails
        sn_host.create_dataset("mag_r", data=np.array([21.0, 25.0]))

        # ---- AGN hosts ---- (present but transients fail cuts; class will be inactive)
        agn_host = ht.create_group("AGN")
        gids_agn = np.array([b"10000001"])
        agn_host.create_dataset("GID", data=gids_agn)
        agn_host.create_dataset("z", data=np.array([0.7]))
        agn_host.create_dataset("a_rot", data=np.array([0.0]))
        for name in ["a0", "b0", "a1", "b1", "ellipticity0", "ellipticity1", "n0", "n1"]:
            agn_host.create_dataset(name, data=np.array([1.0]))
        agn_host.create_dataset("mag_r", data=np.array([30.0]))  # too faint

        # ---- SNII transients ----
        sn_tt = tt.create_group("SNII")

        # Subclass A: 2 rows; first passes (mag_r <= 22), second fails (all 99s -> NaNs)
        A = sn_tt.create_group("A")
        A.create_dataset("z", data=np.array([0.6, 0.4]))
        A.create_dataset("GID", data=np.array([gids_sn[0], gids_sn[1]]))
        A.create_dataset("ra_off", data=np.array([0.01, -0.01]))
        A.create_dataset("dec_off", data=np.array([0.02, -0.02]))
        A.create_dataset("MJD", data=np.tile(np.array([1.0, 2.0, 3.0]), (2, 1)))
        for b in ("u", "g", "r", "i", "z", "Y"):
            if b == "r":
                A.create_dataset("mag_r", data=np.array([[21.0, 22.0, 21.0],
                                                         [99.0, 99.0, 99.0]]))
            else:
                A.create_dataset(f"mag_{b}", data=np.array([[99.0, 99.0, 99.0],
                                                            [99.0, 99.0, 99.0]]))

        # Subclass B: 1 row; passes
        B = sn_tt.create_group("B")
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
        X = agn_tt.create_group("X")
        X.create_dataset("z", data=np.array([0.8]))
        X.create_dataset("GID", data=np.array([gids_agn[0]]))
        X.create_dataset("ra_off", data=np.array([0.0]))
        X.create_dataset("dec_off", data=np.array([0.0]))
        X.create_dataset("MJD", data=np.array([[5.0, 6.0, 7.0]]))
        for b in ("u", "g", "r", "i", "z", "Y"):
            if b == "r":
                X.create_dataset("mag_r", data=np.array([[30.0, 30.0, 30.0]]))  # too faint
            else:
                X.create_dataset(f"mag_{b}", data=np.array([[99.0, 99.0, 99.0]]))

    return path


@pytest.fixture
def scotch_instance(scotch_h5):
    """
    Construct a ScotchSources with deterministic RNG and an r-band cut (<=22).
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    sky_area = 1.0 * u.deg**2
    kwargs_cut = {"band": ["r"], "band_max": [22.0]}
    inst = scotch_module.ScotchSources(
        cosmo=cosmo,
        sky_area=sky_area,
        scotch_path=scotch_h5,
        transient_types=None,
        kwargs_cut=kwargs_cut,
        rng=np.random.default_rng(123),
    )
    return inst

def test_norm_band_names():
    _norm = scotch_module._norm_band_names
    assert _norm(["U", "g", "Y", " y  "]) == ["u", "g", "Y", "Y"]

def test_galaxy_projected_eccentricity_deterministic():
    e1, e2 = scotch_module.galaxy_projected_eccentricity(ellipticity=0.0, rotation_angle=None)
    assert np.isclose(e1, 0.0) and np.isclose(e2, 0.0)

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