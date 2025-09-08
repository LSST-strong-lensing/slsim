import h5py
import pytest
import numpy as np
import astropy.units as u
import slsim.Sources.SourcePopulation.scotch_sources as scotch_module

from pathlib import Path
from astropy.cosmology import FlatLambdaCDM
from slsim.Sources.source import Source
from slsim.Sources.SourceTypes.point_source import PointSource
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
        A = sn_tt.create_group("A")
        A.create_dataset("z", data=np.array([0.6, 0.4]))
        A.create_dataset("GID", data=np.array([gids_sn[0], gids_sn[1]]))
        A.create_dataset("ra_off", data=np.array([0.01, -0.01]))
        A.create_dataset("dec_off", data=np.array([0.02, -0.02]))
        A.create_dataset("MJD", data=np.tile(np.array([1.0, 2.0, 3.0]), (2, 1)))
        for b in ("u", "g", "r", "i", "z", "Y"):
            if b == "r":
                A.create_dataset(
                    "mag_r", data=np.array([[21.0, 22.0, 21.0], [99.0, 99.0, 99.0]])
                )
            else:
                A.create_dataset(
                    f"mag_{b}", data=np.array([[99.0, 99.0, 99.0], [99.0, 99.0, 99.0]])
                )

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


# -----------------------------
# Actual tests
# -----------------------------


def test_norm_band_names():
    _norm = scotch_module._norm_band_names
    assert _norm(["U", "g", "Y", " y  "]) == ["u", "g", "Y", "Y"]


def test_galaxy_projected_eccentricity_deterministic():
    e1, e2 = scotch_module.galaxy_projected_eccentricity(
        ellipticity=0.0, rotation_angle=None
    )
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


def test_host_pass_mask(scotch_instance):
    host_grp = scotch_instance._index["SNII"].host_grp
    mask = scotch_instance._host_pass_mask(host_grp)
    assert mask.dtype == bool
    assert mask.shape == host_grp["z"].shape
    assert mask.tolist() == [True, False]


def test_transient_pass_mask_and_selection(scotch_instance):
    cls = "SNII"
    ci = scotch_instance._index[cls]
    # Subclass "A" -> [True, False]
    subA = next(s for s in ci.subclasses if s.name == "A")
    maskA = scotch_instance._transient_pass_mask(
        subA.grp, ci.host_gid_sorted, ci.host_mask_sorted, batch=1
    )
    assert maskA.tolist() == [True, False]
    # Subclass "B" -> [True]
    subB = next(s for s in ci.subclasses if s.name == "B")
    maskB = scotch_instance._transient_pass_mask(
        subB.grp, ci.host_gid_sorted, ci.host_mask_sorted, batch=1
    )
    assert maskB.tolist() == [True]
    # Totals reflect only active classes with survivors
    assert scotch_instance.source_number == 3  # total rows in SNII (2 + 1)
    assert scotch_instance.source_number_selected == 2  # two survivors


def test_sample_from_class_yields_valid_indices(scotch_instance):
    s, i = scotch_instance._sample_from_class("SNII")
    # With our data: both subclasses only have index 0 eligible
    assert i == 0
    assert s.N >= 1
    assert s.n_ok >= 1


def test_host_lookup(scotch_instance):
    gid = scotch_instance._index["SNII"].host_grp["GID"][0]
    idx = scotch_instance._host_lookup("SNII", gid)
    assert idx == 0
    with pytest.raises(KeyError):
        scotch_instance._host_lookup("SNII", b"99999999")


def test_build_host_dict_and_hostless(scotch_instance):
    host_grp = scotch_instance._index["SNII"].host_grp
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


def test_draw_source(scotch_instance):

    src = scotch_instance.draw_source()
    assert isinstance(src, Source)
    assert isinstance(src._source, PointSource) or isinstance(
        src._source, PointPlusExtendedSource
    )


def test_close(scotch_instance):
    scotch_instance.close()
    assert hasattr(scotch_instance.f, "id")
    assert not scotch_instance.f.id.valid
