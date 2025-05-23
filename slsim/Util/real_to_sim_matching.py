# real_to_sim_matching.py
# ────────────────────────────────────────────────────────────────────
#  Build/clean COSMOS catalogue               ➜ build_real_catalog
#  Build/load SkyPy simulation                ➜ build_sim_catalog
#  Match the two tables                       ➜ build_matched_table
#  Quick scatter-plot helper                  ➜ plot_mag_size
# --------------------------------------------------------------------
from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
import hashlib
import pickle
import numpy as np
import astropy.units as u

from astropy.table import Table, join, hstack
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM


# ╭──────────────────────────────────────────────────────────────────╮
# │  helpers                                                        │
# ╰──────────────────────────────────────────────────────────────────╯
def _fingerprint(*items) -> str:
    """Md5 hash of (files + inputs) –  used for cache key."""
    md5 = hashlib.md5()
    for it in items:
        if isinstance(it, (str, Path)):
            p = Path(it).expanduser().resolve()
            md5.update(str(p).encode())
            if p.exists():
                md5.update(str(p.stat().st_mtime_ns).encode())
        else:
            md5.update(pickle.dumps(it))
    return md5.hexdigest()


def _values(arr):
    """Return plain numpy array, stripping Astropy units if present."""
    try:
        return np.asarray([v.value if hasattr(v, "unit") else v for v in arr])
    except Exception:
        return np.asarray(arr)


def _normalise(x):
    rng = np.max(x) - np.min(x)
    return (x - np.min(x)) / rng if rng != 0 else np.zeros_like(x)


# ╭──────────────────────────────────────────────────────────────────╮
# │  1)  build_real_catalog                                         │
# ╰──────────────────────────────────────────────────────────────────╯
def build_real_catalog(
    *,
    catalog_paths: dict,
    cosmo: FlatLambdaCDM,
    sky_area_deg2: float,
    source_params: dict,
    mag_limit_real: float | None = None,
    size_min_kpc: float | None = None,
) -> Table:
    """Re-implements the “Final_catalog_with_cuts” construction from the
    notebook.

    Only columns needed for matching / plotting are retained.
    """
    cat1 = Table.read(catalog_paths["cat1"], format="fits", hdu=1)
    cat2 = Table.read(catalog_paths["cat2"], format="fits", hdu=1)
    morpho = Table.read(catalog_paths["morpho"], format="ascii")

    cat1["RA"] = cat1["RA"] * u.deg
    cat1["DEC"] = cat1["DEC"] * u.deg
    merged = join(cat1, cat2, keys="IDENT")

    # -------- spatial + mag merge (same as NB) ----------------------
    def merge_catalogs(
        large_cat,
        small_cat,
        lc_mag_col,
        sc_mag_col,
        tolerance=1.0,
        lc_ra_col="RA",
        lc_dec_col="DEC",
        sc_ra_col="RA",
        sc_dec_col="DEC",
    ):
        Lcoords = SkyCoord(ra=large_cat[lc_ra_col], dec=large_cat[lc_dec_col])
        Scoords = SkyCoord(ra=small_cat[sc_ra_col], dec=small_cat[sc_dec_col])
        idx, d2d, _ = Scoords.match_to_catalog_sky(Lcoords)
        mask = d2d.to(u.arcsec) < tolerance * u.arcsec
        small_cat["MAG_DIFF"] = np.nan
        small_cat["MAG_DIFF"][mask] = (
            small_cat[sc_mag_col][mask] - large_cat[lc_mag_col][idx][mask]
        )
        return hstack([large_cat[idx[mask]], small_cat[mask]])

    Final_catalog = merge_catalogs(
        merged, morpho, lc_mag_col="MAG", sc_mag_col="MAG_AUTO_ACS", tolerance=1.0
    )

    # -------- derived quantities ------------------------------------
    ang_dist = cosmo.angular_diameter_distance(Final_catalog["zphot"])
    Final_catalog["RHALFreal"] = (
        Final_catalog["R_HALF"] * Final_catalog["PIXEL_SCALE"] * ang_dist * 4.84814e-3
    )
    Final_catalog["MAGabs"] = (
        Final_catalog["MAG"] + 5 - 5 * np.log10(ang_dist.value * 1e6)
    )

    phi_deg = np.vstack(Final_catalog["sersicfit"])[:, 5]
    Final_catalog["phi_G"] = np.deg2rad(phi_deg)

    # -------- rename + drop -----------------------------------------
    rename = dict(
        RA_1="RA",
        DEC_1="DEC",
        R_PETRO="RPETRO",
        R_HALF="RHALF",
        CONC_PETRO="CONCPETRO",
    )
    for old, new in rename.items():
        if old in Final_catalog.colnames:
            Final_catalog.rename_column(old, new)

    for col in (
        "RA_2",
        "DEC_2",
        "NOISE_MEAN",
        "MAG_AUTO_ACS",
        "fit_status",
        "fit_mad_s",
        "fit_mad_b",
        "fit_dvc_btt",
        "use_bulgefit",
        "hlr",
    ):
        if col in Final_catalog.colnames:
            Final_catalog.remove_column(col)

    # -------- cuts + exclusion list ---------------------------------
    ids_exclude = np.array(source_params.get("source_exclusion_list", []))
    mask = ~np.isin(Final_catalog["IDENT"], ids_exclude)
    if mag_limit_real is not None:
        mask &= Final_catalog["MAGabs"] < mag_limit_real
    if size_min_kpc is not None:
        mask &= Final_catalog["RHALFreal"] > size_min_kpc * u.kpc

    return Final_catalog[mask]


# ╭──────────────────────────────────────────────────────────────────╮
# │  2)  build_sim_catalog                                           │
# ╰──────────────────────────────────────────────────────────────────╯
def build_sim_catalog(
    *,
    sky_area_deg2: float,
    cosmo: FlatLambdaCDM,
    kwargs_cut: dict | None = None,
    source_size: str | None = None,
    skypy_config: str | Path | None = None,
) -> Table:
    """Replicates the notebook’s SkyPy ➜ Galaxies pipeline and returns the
    filtered Astropy table."""
    from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
    from slsim.Sources.galaxies import Galaxies

    sky_area = sky_area_deg2 * u.deg**2
    pipeline = SkyPyPipeline(
        skypy_config=skypy_config,
        sky_area=sky_area,
        filters=None,
        cosmo=cosmo,
    )
    galaxy_list = pipeline.blue_galaxies
    if kwargs_cut is None:
        kwargs_cut = dict(band="i", band_max=20, z_min=0.1, z_max=1.5)

    sim = Galaxies(
        galaxy_list=galaxy_list,
        kwargs_cut=kwargs_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
        source_size=source_size,
    )

    ang_dist = cosmo.angular_diameter_distance(sim._galaxy_select["z"])
    sim._galaxy_select["mag_i_abs"] = np.zeros_like(sim._galaxy_select["mag_i"])
    sim._galaxy_select["mag_i_abs"] = (
        sim._galaxy_select["mag_i"] + 5 - 5 * np.log10((ang_dist.value) * 1e6)
    )
    return sim._galaxy_select.copy()


# ╭──────────────────────────────────────────────────────────────────╮
# │  3)  build_matched_table                                         │
# ╰──────────────────────────────────────────────────────────────────╯
def build_matched_table(
    *,
    catalog_paths: dict,
    cosmo_kwargs: dict,
    sky_area_deg2: float,
    source_params: dict,
    sim_cat_path: str | Path | None = None,
    build_sim_kwargs: dict | None = None,
    mag_limit_real: float | None = None,
    size_min_kpc: float | None = None,
    mag_limit_sim: float | None = None,
    nn_tolerance: float | None = None,
    n_neighbors: int = 1,
    cache: bool = False,
    cache_dir: str | Path = "~/.cache",
    return_tables: bool = False,
):
    """Wrapper that produces (& optionally caches) the full matched table.

    If *return_tables* is True, the tuple (matched, sim_table,
    real_table) is returned instead of only *matched*.
    """
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _fingerprint(
        catalog_paths,
        cosmo_kwargs,
        sky_area_deg2,
        source_params,
        sim_cat_path,
        build_sim_kwargs,
        mag_limit_real,
        size_min_kpc,
        mag_limit_sim,
        nn_tolerance,
        n_neighbors,
    )
    cache_file = cache_dir / f"matched_{tag}.pkl"
    if cache and cache_file.exists():
        matched, sim_tab, real_tab = pickle.loads(cache_file.read_bytes())
        return (matched, sim_tab, real_tab) if return_tables else matched

    cosmo = FlatLambdaCDM(**cosmo_kwargs)

    real_tab = build_real_catalog(
        catalog_paths=catalog_paths,
        cosmo=cosmo,
        sky_area_deg2=sky_area_deg2,
        source_params=source_params,
        mag_limit_real=mag_limit_real,
        size_min_kpc=size_min_kpc,
    )

    if sim_cat_path is not None:
        sim_tab = Table.read(sim_cat_path, format="fits")
    else:
        if build_sim_kwargs is None:
            raise ValueError("Provide either `sim_cat_path` or `build_sim_kwargs`")
        sim_tab = build_sim_catalog(
            sky_area_deg2=sky_area_deg2, cosmo=cosmo, **build_sim_kwargs
        )

    if mag_limit_sim is not None and "mag_i" in sim_tab.colnames:
        sim_tab = sim_tab[sim_tab["mag_i"] < mag_limit_sim]

    # ----- perform match (slsim’s utility) ---------------------------
    from slsim.Util.cosmo_util import match_simulated_to_real

    matched = match_simulated_to_real(
        sim_table=sim_tab,
        cosmos_catalog=real_tab,
        cosmo=cosmo,
        tolerance=nn_tolerance,
        n_neighbors=n_neighbors,
    )

    if cache:
        cache_file.write_bytes(pickle.dumps((matched, sim_tab, real_tab)))

    return (matched, sim_tab, real_tab) if return_tables else matched


# ╭──────────────────────────────────────────────────────────────────╮
# │  4)  scatter-plot helper                                         │
# ╰──────────────────────────────────────────────────────────────────╯
def plot_mag_size(
    real_table: Table,
    sim_table: Table,
    *,
    mag_real: str = "MAGabs",
    size_real: str = "RHALFreal",
    mag_sim: str = "mag_i_abs",
    size_sim: str = "physical_size",
    ax=None,
    show: bool = True,
    save: str | Path | None = None,
    **scatter_kwargs,
):
    """Quick visual comparison in (absolute magnitude, half-light size) space.

    Parameters
    ----------
    real_table / sim_table
        Astropy tables as returned by `build_real_catalog` / `build_sim_catalog`.
    mag_real / size_real
        Column names in *real_table*.
    mag_sim / size_sim
        Column names in *sim_table*.
    ax
        Existing matplotlib Axes (created if None).
    show
        Whether to call `plt.show()` (ignored if *ax* provided).
    save
        Optional file path to `fig.savefig(...)`.
    **scatter_kwargs
        Extra kwargs forwarded to `ax.scatter` (affect both clouds).
    """
    if "mag_i_abs" not in sim_table.colnames and "mag_i" in sim_table.colnames:
        ang = FlatLambdaCDM(H0=70, Om0=0.3).angular_diameter_distance(sim_table["z"])
        sim_table["mag_i_abs"] = sim_table["mag_i"] + 5 - 5 * np.log10(ang.value * 1e6)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # handle units
    x_real = _values(real_table[size_real])
    y_real = _values(real_table[mag_real])
    x_sim = _values(sim_table[size_sim])
    y_sim = _values(sim_table[mag_sim])

    # default styles
    kw_real = dict(s=8, color="tab:blue", alpha=0.6, label="COSMOS real")
    kw_sim = dict(s=8, color="tab:orange", alpha=0.6, label="SkyPy sim")
    kw_real.update(scatter_kwargs)
    kw_sim.update(scatter_kwargs)

    ax.scatter(x_real, y_real, **kw_real)
    ax.scatter(x_sim, y_sim, marker="x", **kw_sim)

    ax.set_xlabel(
        "Half-light radius  [kpc]" if size_real.endswith("real") else size_real
    )
    ax.set_ylabel("Absolute magnitude $M_i$")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.set_title("Magnitude-size plane")

    if save is not None:
        Path(save).expanduser().parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=300, bbox_inches="tight")
    if show and ax is None:
        plt.show()
    return ax
