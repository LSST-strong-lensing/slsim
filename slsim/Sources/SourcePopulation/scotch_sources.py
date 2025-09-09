import h5py
import warnings

import numpy as np

from typing import Callable
from scipy.integrate import quad
from slsim.Util import param_util
from dataclasses import dataclass
from slsim.Sources.source import Source
from astropy.cosmology import Cosmology
from slsim.Sources.SourcePopulation.source_pop_base import SourcePopBase

BANDS = ("u", "g", "r", "i", "z", "Y")
SCOTCH_MAPPINGS = {
    "n0": "n_sersic_0",
    "n1": "n_sersic_1",
    "e0": "ellipticity0",
    "e1": "ellipticity1",
}


def d08(z: float | np.ndarray) -> float | np.ndarray:
    """Redshift Evolution of SNIa Rates from Dilday et al.

    2008 Sec. 6.4.1
    https://arxiv.org/abs/0801.3297
    """

    return (1 + z) ** 1.5


def md14(z: float | np.ndarray) -> float | np.ndarray:
    """Redshift Evolution of Cosmic Star Formation Rate from Madau & Dickinson
    2014 Eq.

    15.
    https://arxiv.org/abs/1403.0007
    """

    return (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)


def s15(z: float | np.ndarray) -> float | np.ndarray:
    """Redshift Evolution of CCSNe Rates from Strolger et al.

    2015 Eq. 9.
    https://arxiv.org/abs/1509.06574
    """

    return (1 + z) ** 5.0 / (1 + ((1 + z) / 1.5) ** 6.1)


def snia_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 25  # in units of 10^-6 Mpc^-3 yr^-1
    z = np.asarray(z)
    rate = np.where(z < 1, r0 * d08(z), r0 * (1 + z) ** -0.5)

    return rate


def snia_91bg_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 3  # in units of 10^-6 Mpc^-3 yr^-1
    z = np.asarray(z)
    rate = r0 * d08(z)

    return rate


def sniax_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 6
    z = np.asarray(z)
    rate = r0 * md14(z)

    return rate


def snii_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 45  # in units of 10^-6 Mpc^-3 yr^-1
    z = np.asarray(z)
    rate = r0 * s15(z)

    return rate


def snibc_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 19  # in units of 10^-6 Mpc^-3 yr^-1
    z = np.asarray(z)
    rate = r0 * s15(z)

    return rate


def slsn_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 0.02  # in units of 10^-6 Mpc^-3 yr^-1
    z = np.asarray(z)
    rate = r0 * md14(z)

    return rate


def tde_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 1  # in units of 10^-6 Mpc^-3 yr^-1
    z = np.asarray(z)
    rate = r0 * 10 ** (-5 * z / 6)

    return rate


def kn_rate(z: float | np.ndarray) -> float | np.ndarray:

    r0 = 6
    z = np.asarray(z)
    rate = r0 * np.ones_like(z)

    return rate


# Subclass rates are calculated using Lokken et al. 2022
# Table B1 if not given in Kessler et al. 2019
# https://arxiv.org/abs/1903.11756 Table 2
RATE_FUNCS = {
    "SNIa-SALT2": snia_rate,
    "SNIax": sniax_rate,
    "SNIa-91bg": snia_91bg_rate,
    "SNII-Templates": lambda z: 0.19448 * snii_rate(z),
    "SNII-NMF": lambda z: 0.19948 * snii_rate(z),
    "SNII+HostXT_V19": lambda z: 0.39016 * snii_rate(z),
    "SNIIn+HostXT_V19": lambda z: 0.04502 * snii_rate(z),
    "SNIIn-MOSFIT": lambda z: 0.04502 * snii_rate(z),
    "SNIb-Templates": lambda z: 0.27835 * snibc_rate(z),
    "SNIb+HostXT_V19": lambda z: 0.27835 * snibc_rate(z),
    "SNIc-Templates": lambda z: 0.19330 * snibc_rate(z),
    "SNIc+HostXT_V19": lambda z: 0.19330 * snibc_rate(z),
    "SNIcBL+HostXT_V19": lambda z: 0.05670 * snibc_rate(z),
    "SNIIb+HostXT_V19": lambda z: 0.13085 * snii_rate(z),
    "SLSN-I": slsn_rate,
    "KN_K17": lambda z: 0.5 * kn_rate(z),
    "KN_B19": lambda z: 0.5 * kn_rate(z),
    "TDE": tde_rate,
}


def expected_number(
    rate_fn: Callable,
    cosmo: Cosmology,
    z_min: float = 0.0,
    z_max: float = 3.0,
) -> float:

    def integrand(z):

        dv = 4 * np.pi * cosmo.differential_comoving_volume(z).value
        volumetric_rate = 1e-6 * rate_fn(z)

        return volumetric_rate * dv

    n = quad(integrand, z_min, z_max)[0]

    return n


def _norm_band_names(bands: list[str]) -> list[str]:
    """Normalize band names to lowercase, except for 'Y' which is uppercase.

    Parameters
    ----------
    bands : list of str
        List of band names to normalize.

    Returns
    -------
    list of str
        Normalized band names.
    """

    out = []
    for b in bands:
        b = b.strip()
        if b.lower() == "y":
            out.append("Y")
        else:
            out.append(b.lower())
    return out


def galaxy_projected_eccentricity(
    ellipticity: float, rotation_angle=float | None
) -> tuple[float, float]:
    """Compute the projected eccentricity components (e1, e2) of an elliptical
    galaxy given its ellipticity and rotation angle. If the rotation angle is
    not provided, it is drawn randomly from a uniform distribution between 0
    and π.

    Parameters
    ----------
    ellipticity : float
        Eccentricity amplitude, must be in the range [0, 1).
    rotation_angle : float or None, optional
        Rotation angle of the major axis in radians. The reference is the +RA axis
        (towards the East direction) and it increases from East to North. If None, a random angle is drawn.

    Returns
    -------
    e1 : float
        First component of the projected eccentricity.
    e2 : float
        Second component of the projected eccentricity.
    """

    if rotation_angle is None:
        phi = np.random.uniform(0, np.pi)
    else:
        phi = rotation_angle
    e = param_util.epsilon2e(ellipticity)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2


@dataclass
class _SubclassIndex:
    name: str
    grp: h5py.Group
    N: int
    n_ok: int
    n_expected: int
    eligible: np.ndarray | None  # None => all rows valid


@dataclass
class _ClassIndex:
    host_grp: h5py.Group
    host_gid_sorted: np.ndarray
    host_gid_sort_idx: np.ndarray
    host_mask_sorted: np.ndarray  # boolean, aligned with host_gid_sorted
    subclasses: list[_SubclassIndex]
    subclass_total: np.ndarray
    subclass_expected: np.ndarray
    subclass_selected: np.ndarray
    total: int
    total_expected: int
    total_selected: int = 0


class ScotchSources(SourcePopBase):
    def __init__(
        self,
        cosmo: Cosmology,
        scotch_path: str,
        sky_area=None,
        transient_types: list[str] | None = None,
        kwargs_cut: dict | None = None,
        rng: np.random.Generator | int | None = None,
    ):
        """Class for SCOTCH transient source population. Allows for sampling of
        transients and their hosts from the SCOTCH HDF5 catalogs.

        Parameters
        ----------
        cosmo : astropy.cosmology instance
            An instance of an astropy cosmology model (e.g., FlatLambdaCDM(H0=70, Om0=0.3)).
        scotch_path : str
            Path to the SCOTCH HDF5 file.
        sky_area : astropy.units.Quantity, optional
            Sky area over which galaxies are sampled. Must be in units of solid angle.
            Default is None.
        transient_types : list of str, optional
            List of transient types to include. If None, all available types are used.
            Default is None.
        kwargs_cut : dict, optional
            Dictionary of selection criteria to filter the sources. Supported keys:
            - 'z_min': Minimum redshift (float).
            - 'z_max': Maximum redshift (float).
            - 'band': List of band names (str) for magnitude cuts.
            - 'band_max': List of maximum magnitudes (float) corresponding to 'band'.
            The lengths of 'band' and 'band_max' must be equal. Default is None
        rng : np.random.Generator, int, or None, optional
            Random number generator or seed for reproducibility. If None, a new
            generator is created. Default is None.
        Raises
        ------
        ValueError
            If transient_types contains unknown types, or if kwargs_cut is invalid,
            or if no sources pass the selection criteria.
        Warnings
            If any transient class has no objects passing the provided kwargs_cut filters.
        Notes
        -----
        The SCOTCH catalogs contain multiple transient classes, each with its own
        host galaxy table. Transients are sampled uniformly among the selected classes
        and subclasses, with selection cuts applied as specified in kwargs_cut.
        Hosts are included if their redshift is not 999.0; otherwise, the transient
        is considered hostless.

        The transient lightcurves are provided as "general_lightcurve" point sources,
        and hosts (if any) as "double_sersic" extended sources. If the
        transient is hostless, the Source is an instance of PointSource; otherwise,
        it is an instance of PointPlusExtendedSource.

        The SCOTCH HDF5 file is expected to have the following structure:
        - /TransientTable/{transient_class}/{subclass}/
            - Datasets: "z", "GID", "ra_off", "dec_off", "MJD", "mag_{band}" for each band
        - /HostTable/{transient_class}/
            - Datasets: "GID", "z", "mag_{band}" for each band, "a_rot", "a0", "b0", "n
            "ellipticity0", "a1", "b1", "n1", etc.
        The "GID" fields are used to link transients to their hosts.
        The "mag_{band}" datasets contain magnitudes, with 99.0 indicating missing data.
        """

        super().__init__(cosmo=cosmo, sky_area=sky_area)
        self.f = h5py.File(scotch_path, "r")

        # Parse transient types
        avail = set(self.f["TransientTable"].keys())
        if transient_types is None:
            transient_types = sorted(avail)
        else:
            missing = [t for t in transient_types if t not in avail]
            if missing:
                raise ValueError(
                    f"Unknown transient_types {missing}. Available: {sorted(avail)}"
                )
        self.transient_types = list(transient_types)

        # Parse kwargs_cut
        self.bands, self.band_max = [], []
        z_min = kwargs_cut.get("z_min") if kwargs_cut else None
        z_max = kwargs_cut.get("z_max") if kwargs_cut else None
        if kwargs_cut and ("band" in kwargs_cut or "band_max" in kwargs_cut):
            band = kwargs_cut.get("band")
            band_max = kwargs_cut.get("band_max")
            band_is_str = isinstance(band, str)
            bandmax_is_num = isinstance(band_max, (int, float))

            if band_is_str and bandmax_is_num:
                kwargs_cut["band"] = [band]
                kwargs_cut["band_max"] = [band_max]

            band = kwargs_cut.get("band")
            band_max = kwargs_cut.get("band_max")

            band_is_list = isinstance(band, (list, tuple))
            bandmax_is_list = isinstance(band_max, (list, tuple))
            band_and_bandmax_equal_len = len(band) == len(band_max)
            is_valid = band_is_list and bandmax_is_list and band_and_bandmax_equal_len

            if not is_valid:
                raise ValueError(
                    "kwargs_cut['band'] and ['band_max'] must be lists of equal length."
                )
            self.bands = _norm_band_names(list(band))
            self.band_max = list(map(float, band_max))
            for b in self.bands:
                if b not in BANDS:
                    raise ValueError(f"Unsupported band '{b}'. Allowed: {BANDS}")

        self.zmin = 0.0 if z_min is None else float(z_min)
        self.zmax = 3.0 if z_max is None else float(z_max)  # Max in SCOTCH

        self.rng = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )

        # Build indices per class
        self._index: dict[str, _ClassIndex] = {}
        for cls in self.transient_types:
            # Hosts: precompute sorted GIDs and a boolean mask for host filters
            host_grp = self.f["HostTable"][cls]
            host_gids = host_grp["GID"][...]  # |S8
            sort_idx = np.argsort(host_gids)
            gids_sorted = host_gids[sort_idx]

            host_mask = self._host_pass_mask(host_grp)  # (Nh,)
            host_mask_sorted = host_mask[sort_idx]

            # Transient subclasses: build eligible lists with chunked scans
            sub_list: list[_SubclassIndex] = []
            subclass_total = []
            subclass_expected = []
            subclass_selected = []
            for subname, subgrp in self.f["TransientTable"][cls].items():

                eligible_mask = self._transient_pass_mask(
                    subgrp, gids_sorted, host_mask_sorted
                )
                n_ok = int(eligible_mask.sum())
                if n_ok == 0:
                    continue
                N = eligible_mask.size
                eligible_idx = (
                    None
                    if n_ok == N
                    else np.flatnonzero(eligible_mask).astype(np.int64)
                )

                if subname in RATE_FUNCS:
                    rate_fn = RATE_FUNCS[subname]
                    n_expected = int(
                        expected_number(
                            rate_fn=rate_fn,
                            cosmo=cosmo,
                            z_min=self.zmin,
                            z_max=self.zmax,
                        )
                    )
                elif "AGN" in subname:
                    n_expected = n_ok
                else:
                    raise ValueError(
                        f"Transient Subclass {subname} not found in rate functions. "
                        + f"Rate functions are available for {list(RATE_FUNCS.keys())}."
                    )

                sub_list.append(
                    _SubclassIndex(
                        name=subname,
                        grp=subgrp,
                        N=N,
                        n_ok=n_ok,
                        n_expected=n_expected,
                        eligible=eligible_idx,
                    )
                )

                subclass_total.append(N)
                subclass_selected.append(n_ok)
                subclass_expected.append(n_expected)

            subclass_total = np.asarray(subclass_total)
            subclass_selected = np.asarray(subclass_selected)
            subclass_expected = np.asarray(subclass_expected)
            total = np.sum(subclass_total)
            total_selected = np.sum(subclass_selected)
            total_expected = np.sum(subclass_expected)

            self._index[cls] = _ClassIndex(
                host_grp=host_grp,
                host_gid_sorted=gids_sorted,
                host_gid_sort_idx=sort_idx,
                host_mask_sorted=host_mask_sorted,
                subclasses=sub_list,
                subclass_total=subclass_total,
                subclass_selected=subclass_selected,
                subclass_expected=subclass_expected,
                total=total,
                total_expected=total_expected,
                total_selected=total_selected,
            )

        # keep only classes with survivors
        active_types = []
        total = 0
        total_selected = 0
        total_expected = 0
        for c in self.transient_types:
            if self._index[c].total_selected > 0:
                active_types.append(c)
                total += self._index[c].total
                total_selected += self._index[c].total_selected
                total_expected += self._index[c].total_expected
            else:
                warnings.warn(
                    f"Transient class '{c}' has no objects passing "
                    + "the provided kwargs_cut filters and will be ignored.",
                )

        self.n_source = total
        self.n_source_selected = total_selected
        self.total_expected = total_expected
        self.active_transient_types = active_types

        if self.n_source_selected == 0:
            raise ValueError("No objects satisfy the provided kwargs_cut filters.")

    @property
    def source_number(self) -> int:
        """Number of sources in the population before any selection cuts.

        Returns
        -------
        int
            Number of sources.
        """
        return self.n_source

    @property
    def source_number_selected(self) -> int:
        """Number of sources in the population after applying selection cuts.

        Returns
        -------
        int
            Number of sources passing the selection criteria.
        """
        return self.n_source_selected

    # -------------------- filtering helpers --------------------

    def _host_pass_mask(self, host_grp: h5py.Group) -> np.ndarray:
        """Create a boolean mask for hosts passing cuts on redshift and
        magnitude.

        Parameters
        ----------
        host_grp : h5py.Group
            HDF5 group for the host table of a given transient class. Must contain
            datasets "z" and "mag_{band}" for each band in self.bands, all of shape (Nh,).

        Returns
        -------
        mask : np.ndarray
            Boolean array with shape (Nh,) where True indicates the host passes all cuts.
        """
        Nh = host_grp["z"].shape[0]
        mask = np.ones(Nh, dtype=bool)

        z = host_grp["z"][...]
        is_hostless = z == 999.0
        passes_redshift_cut = (z >= self.zmin) & (z <= self.zmax)
        mask &= np.isfinite(z) & (is_hostless | passes_redshift_cut)

        for b, mmax in zip(self.bands, self.band_max):
            arr = host_grp[f"mag_{b}"][...]
            mask &= np.isfinite(arr) & (arr <= mmax)

        return mask

    def _transient_pass_mask(
        self,
        subgrp: h5py.Group,
        host_gid_sorted: np.ndarray,
        host_mask_sorted: np.ndarray,
        batch: int = 100_000,
    ) -> np.ndarray:
        """Create a boolean mask for transients passing cuts on redshift,
        magnitude, and host validity. Lightcurve magnitude cuts are applied as
        nanmin over time <= threshold.

        Parameters
        ----------
        subgrp : h5py.Group
            HDF5 group for a transient subclass. Must contain datasets "z", "GID", and
            "mag_{band}" for each band in self.bands, where "z" and "GID" have shape (N,)
            and "mag_{band}" has shape (N, T).
        host_gid_sorted : np.ndarray
            Sorted array of host GIDs (|S8) for the corresponding transient class.
        host_mask_sorted : np.ndarray
            Boolean array aligned with host_gid_sorted indicating valid hosts.
        batch : int, optional
            Number of transient rows to process in each chunk, by default 100_000.

        Returns
        -------
        mask : np.ndarray
            Boolean array with shape (N,) where True indicates the transient passes all cuts.
        """
        N = subgrp["z"].shape[0]
        mask = np.ones(N, dtype=bool)

        # transient redshift
        z = subgrp["z"][...]
        mask &= np.isfinite(z) & (z >= self.zmin) & (z <= self.zmax)

        # transient bands: require nanmin over time <= threshold for each requested band
        for b, mmax in zip(self.bands, self.band_max):
            ds = subgrp[f"mag_{b}"]  # shape (N, T)
            # chunk along rows
            for i in range(0, N, batch):
                sl = slice(i, min(i + batch, N))
                arr = ds[sl]  # (B,T)
                # nanmin across time; if all NaN, result is NaN (treated as fail)
                with np.errstate(invalid="ignore"):
                    mn = np.nanmin(arr, axis=1)  # (B,)
                ok = np.isfinite(mn) & (mn <= mmax)
                mask[sl] &= ok

        # host pass via GID membership (vectorized searchsorted per chunk)
        for i in range(0, N, batch):
            sl = slice(i, min(i + batch, N))
            gids = subgrp["GID"][sl]
            pos = np.searchsorted(host_gid_sorted, gids)
            in_range = pos < host_gid_sorted.size
            match = in_range & (host_gid_sorted[pos] == gids)
            host_ok = np.zeros(sl.stop - sl.start, dtype=bool)
            host_ok[match] = host_mask_sorted[pos[match]]
            mask[sl] &= host_ok

        return mask

    # -------------------- sampling --------------------

    def _sample_from_class(self, cls: str) -> tuple[_SubclassIndex, int]:
        """Sample a transient subclass and an index within that subclass,
        uniformly over all surviving objects in the class.

        Parameters
        ----------
        cls : str
            Transient class name.

        Returns
        -------
        s : _SubclassIndex
            The sampled transient subclass.
        i : int
            Index within the subclass's dataset.
        """
        ci = self._index[cls]
        weights = np.array([s.n_ok for s in ci.subclasses], dtype=float)
        weights /= weights.sum()
        s = ci.subclasses[self.rng.choice(len(ci.subclasses), p=weights)]
        if s.eligible is None:
            i = int(self.rng.integers(0, s.N))
        else:
            i = int(s.eligible[self.rng.integers(0, len(s.eligible))])
        return s, i

    def _host_lookup(self, cls: str, gid_bytes: bytes) -> int:
        """Given a transient class and a GID (as bytes), return the index of
        the corresponding host in the HostTable for that class.

        Parameters
        ----------
        cls: str
            Transient class name.
        gid_bytes: bytes
            GID of the host as bytes (|S8).

        Returns
        -------
        int
            Index of the host in the HostTable for the given class.
        """
        ci = self._index[cls]
        pos = int(np.searchsorted(ci.host_gid_sorted, gid_bytes))
        if pos >= len(ci.host_gid_sorted) or ci.host_gid_sorted[pos] != gid_bytes:
            raise KeyError(f"GID {gid_bytes!r} not found in HostTable/{cls}")
        return int(ci.host_gid_sort_idx[pos])

    def _scotch_to_slsim_host(self, host: dict) -> dict:
        """Convert a host dictionary from SCOTCH naming and conventions to
        slsim naming and conventions. Adds projected eccentricity components
        and average angular size for each component.

        Parameters
        ----------
        host : dict
            Dictionary with host parameters using SCOTCH naming. Must include
            keys "ellipticity0", "ellipticity1", "a_rot", "a0", "b0", "a1", "b1".

        Returns
        -------
        dict
            Dictionary with host parameters using slsim naming, including
            "e0_1", "e0_2", "angular_size_0", "e1_1", "e1_2", "angular_size_1".
        """

        _host = host.copy()
        host = _host
        for comp in [0, 1]:

            ellip = host[f"ellipticity{comp}"]
            a_rot = host["a_rot"]
            a = host[f"a{comp}"]
            b = host[f"b{comp}"]

            e1, e2 = galaxy_projected_eccentricity(
                ellipticity=ellip, rotation_angle=a_rot
            )
            angular_size = param_util.average_angular_size(a=a, b=b)

            host[f"e{comp}_1"] = e1
            host[f"e{comp}_2"] = e2
            host[f"angular_size_{comp}"] = angular_size

        return host

    def _build_host_dict(self, host_grp: h5py.Group, host_idx: int) -> dict:
        """Build an SLSlim-compatible host dictionary from the host group and
        index. If the host redshift is 999.0 (corresponding to a hostless
        transient), return an empty dictionary.

        Parameters
        ----------
        host_grp : h5py.Group
            HDF5 group for the host table of a given transient class.
        host_idx : int
            Index of the host within the host group.
        Returns
        -------
        dict
            Dictionary with host parameters using slsim naming. Empty if
            the transient is hostless (host redshift = 999.0).
        """

        host = {}
        if host_grp["z"][host_idx] == 999.0:
            return host

        for name, ds in host_grp.items():

            if not isinstance(ds, h5py.Dataset):
                continue

            val = ds[host_idx]
            if ds.dtype.kind == "S":
                val = val.decode("utf-8")
            if name == "a_rot":
                val = np.deg2rad(val)

            if name in SCOTCH_MAPPINGS:
                name = SCOTCH_MAPPINGS[name]
            host[name] = val

        host = self._scotch_to_slsim_host(host)

        return host

    def _draw_source_dict(self, *args, **kwargs) -> dict:
        """Draw a transient and its host (if any), returning a combined
        dictionary of parameters. Transient class is chosen uniformly among
        those with surviving objects, then a transient is chosen uniformly
        among all surviving subclasses in that class.

        Returns
        -------
        dict
            Dictionary with transient and host parameters using slsim naming.
        bool
            True if the transient has a host, False if hostless.
        """
        cls = self.rng.choice(self.active_transient_types)
        s, i = self._sample_from_class(cls)
        g = s.grp

        transient_metadata = {
            "name": f"{cls}_{s.name}",
            "z": float(g["z"][i]),
            "ra_off": float(g["ra_off"][i]),
            "dec_off": float(g["dec_off"][i]),
        }

        transient_lightcurve = {"MJD": g["MJD"][i]}
        for band in BANDS:
            mags = g[f"mag_{band}"][i]
            mags = np.where(mags == 99.0, np.nan, mags)
            transient_lightcurve[f"ps_mag_{band}"] = mags
        transient_dict = transient_metadata | transient_lightcurve

        gid_b = g["GID"][i]
        host_idx = self._host_lookup(cls, gid_b)
        host_grp = self._index[cls].host_grp
        host_dict = self._build_host_dict(host_grp, host_idx)
        has_host = bool(host_dict)

        source_dict = transient_dict | host_dict

        return source_dict, has_host

    def draw_source(self, *args, **kwargs) -> Source:
        """Draw a source from the population, returning a Source object.
        Transients are instantiated as "general_lightcurve" point sources, and
        hosts (if any) as "double_sersic" extended sources. If the transient is
        hostless, the Source is an instance of PointSource, otherwise it is an
        instance of PointPlusExtendedSource.

        Returns
        -------
        Source
            The drawn source object. If hostless, an instance of PointSource;
            otherwise, an instance of PointPlusExtendedSource.
        """

        source_dict, has_host = self._draw_source_dict()
        point_source_type = "general_lightcurve"
        extended_source_type = "double_sersic"
        if not has_host:
            extended_source_type = None

        source = Source(
            cosmo=self._cosmo,
            extended_source_type=extended_source_type,
            point_source_type=point_source_type,
            **source_dict,
        )

        return source

    def close(self):
        """Close the underlying HDF5 file."""
        try:
            self.f.close()
        except Exception:
            pass
