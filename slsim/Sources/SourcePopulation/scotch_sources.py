import h5py
import warnings

import numpy as np
import astropy.units as u

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
SKY_AREA = (4 * np.pi * u.rad**2).to(u.deg**2).value


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
class _SubclassShard:
    file_index: int
    grp: h5py.Group
    N: int
    n_ok: int
    eligible: np.ndarray | int  # If int then eligible = n_ok = N, so all rows valid
    weight_sum: float  # S_{f,rl} = sum d_{rl}(z_i) over eligible rows
    weights: np.ndarray | None  # Normalized weights over eligible rows


@dataclass
class _SubclassIndex:
    name: str
    shards: list[_SubclassShard]
    n_expected: int  # from RATE_FUNCS integral over z


@dataclass
class _ClassIndex:
    # One host table per input file for this class
    host_grp: list[h5py.Group]
    host_gid_sorted: list[np.ndarray]
    host_gid_sort_idx: list[np.ndarray]
    host_mask_sorted: list[np.ndarray]

    # Per-subclass info (merged across files)
    subclasses: list[_SubclassIndex]
    subclass_total: np.ndarray  # total rows per subclass (sum across files)
    subclass_selected: np.ndarray  # eligible rows per subclass (sum across files)
    subclass_expected: np.ndarray  # expected counts per subclass (RATE_FUNCS)
    subclass_weights: np.ndarray  # sampling weights p(s | class)

    total: int
    total_expected: int
    total_selected: int = 0


class ScotchSources(SourcePopBase):
    def __init__(
        self,
        cosmo: Cosmology,
        scotch_path: list[str] | str,
        sky_area=None,
        transient_types: list[str] | str | None = None,
        transient_subtypes: dict[list[str]] | None = None,
        kwargs_cut: dict | None = None,
        rng: np.random.Generator | int | None = None,
        sample_uniformly: bool = False,
        exclude_agn: bool = False,
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
        transient_subtypes: dict of list of str, optional
            Dict with transient types as keys and lists of transient subtypes to include.
            If None, all available subtypes for chosen transient types are used. Default
            is None.
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
        sample_uniformly: bool, optional
            If False, sampling is done according to the expected rates of transient
            subclasses within the given redshift range. If True, sampling is done
            uniformly over all transient subclasses, while the redshift of the
            transient is still sampled according to the volumetric rate. Default is False.
        exclude_agn: bool, optional
            If True, AGN are excluded from the source population. Defualt is False.
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

        self.files = (
            [h5py.File(p, "r") for p in scotch_path]
            if isinstance(scotch_path, (list, tuple))
            else [h5py.File(scotch_path, "r")]
        )

        self.sample_uniformly = sample_uniformly

        transient_types = self._parse_transient_types(transient_types)
        if "AGN" in transient_types and exclude_agn:
            transient_types = [
                transient_type
                for transient_type in transient_types
                if transient_type != "AGN"
            ]
        self.transient_types = transient_types
        self.transient_subtypes = self._parse_transient_subtypes(transient_subtypes)

        zmin, zmax, bands_to_filter, band_maxes = self._parse_kwargs_cut(kwargs_cut)
        self.zmin, self.zmax = zmin, zmax
        self.bands_to_filter = bands_to_filter
        self.band_maxes = band_maxes

        self.rng = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )

        # Build indices per class

        self._index: dict[str, _ClassIndex] = {}
        for transient_type in self.transient_types:
            self._index[transient_type] = self._build_transient_index(
                transient_type=transient_type
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

        # Setup weights for sampling
        n_active_transient_types = len(self.active_transient_types)
        class_weights = np.zeros(n_active_transient_types)

        for i, c in enumerate(self.active_transient_types):

            cls = self._index[c]
            subclass_expected = cls.subclass_expected

            if sample_uniformly:
                n_subclasses = len(subclass_expected)
                class_weight = n_subclasses
                subclass_weights = np.ones(n_subclasses) / n_subclasses

            else:
                # The probability of sampling a transient class c and a subclass
                # s are given as
                # p(c, s) = n^{expected}_{c,s} / n^{expected}_total.
                # We factorize this such that we first sample the class c with
                # probabilities
                # p(c) = \sum_{s} p(c,s).
                # Given a transient class, we then sample the subclass as
                # p(s | c) = p(c, s) / p(c).
                # Thus p(c, s) = p(s | c) * p(c)

                global_subclass_weights = subclass_expected / self.total_expected
                class_weight = np.sum(global_subclass_weights)
                subclass_weights = global_subclass_weights / class_weight

            cls.subclass_weights = subclass_weights
            class_weights[i] = class_weight

        if sample_uniformly:
            class_weights = class_weights / np.sum(class_weights)
        self.class_weights = class_weights

        self._effective_sky_area = (
            SKY_AREA * self.n_source_selected / self.total_expected
        )
        if self.sky_area is None:
            self.sky_area = self._effective_sky_area * u.deg**2
        else:
            scaling_factor = (self.sky_area / self._effective_sky_area).value
            new_number_selected = int(scaling_factor * self.source_number_selected)
            self.n_source_selected = new_number_selected

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

        # Why not just rename self.n_source_selected as
        # self.source_number_selected. Would mean having to refactor
        # SourcePopBase and any children, but would reduce boat.
        # Grumble grumble grumble
        return self.n_source_selected

    # -------------------- init helpers --------------------

    def _parse_transient_types(self, transient_types: list[str] | str | None) -> list:

        if isinstance(transient_types, str):
            transient_types = [transient_types]

        avail = set()
        for f in self.files:
            avail |= set(f["TransientTable"].keys())
        if transient_types is None:
            transient_types = avail
        else:
            missing = [t for t in transient_types if t not in avail]
            if missing:
                raise ValueError(
                    f"Unknown transient_types {missing}. Available: {sorted(avail)}"
                )
        transient_types = sorted(list(transient_types))

        return transient_types

    def _parse_transient_subtypes(
        self, transient_subtypes: dict | dict[list[str]] | None
    ) -> dict[list[str]] | None:

        if transient_subtypes is None:
            transient_subtypes = {}
        for transient_type in self.transient_types:

            sub_union = set()
            for f in self.files:
                if transient_type in f["TransientTable"]:
                    sub_union |= set(f["TransientTable"][transient_type].keys())

            provided = transient_subtypes.get(transient_type, None)

            if provided is None:
                transient_subtypes[transient_type] = sorted(list(sub_union))
                continue

            missing = [t for t in provided if t not in sub_union]
            if missing:
                raise ValueError(
                    f"Unknown transient_subtypes {missing} for transient_type {transient_type}. "
                    f"Available: {sorted(sub_union)}"
                )

        return transient_subtypes

    def _parse_kwargs_cut(
        self, kwargs_cut: dict | None
    ) -> tuple[float, float, list, list]:

        if kwargs_cut is None:
            kwargs_cut = {}

        z_min = float(kwargs_cut.get("z_min", 0.0))
        z_max = float(kwargs_cut.get("z_max", 3.0))
        bands = []
        band_maxes = []

        has_bands = "band" in kwargs_cut
        has_band_max = "band_max" in kwargs_cut

        if (has_bands and not has_band_max) or (has_band_max and not has_bands):
            raise ValueError(
                'If "band" is provided in kwargs_cut then "band_max" must also be '
                + "provided, and vice versa. Currently provided keys in kwargs_cut"
                + f" are {list(kwargs_cut.keys())}."
            )

        if has_bands and has_band_max:

            band = kwargs_cut.get("band")
            band_max = kwargs_cut.get("band_max")
            band_is_str = isinstance(band, str)
            bandmax_is_num = isinstance(band_max, (int, float))

            if band_is_str:
                kwargs_cut["band"] = [band]
            if bandmax_is_num:
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
            bands = _norm_band_names(list(band))
            band_maxes = list(map(float, band_max))

            for b in bands:
                if b not in BANDS:
                    raise ValueError(f"Unsupported band '{b}'. Allowed: {BANDS}")

        return z_min, z_max, bands, band_maxes

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

        for b, mmax in zip(self.bands_to_filter, self.band_maxes):
            arr = host_grp[f"mag_{b}"][...]
            mask &= np.isfinite(arr) & (arr <= mmax)

        return mask

    def _build_index_host_info(
        self, transient_type: str
    ) -> tuple[list, list, list, list]:
        # Per-file host info
        host_grps = []
        gids_sorted_list = []
        sort_idx_list = []
        host_mask_sorted_list = []

        for f in self.files:
            if transient_type not in f["HostTable"]:
                # If a file lacks this class, create empty stubs to keep indexing aligned
                host_grps.append(None)
                gids_sorted_list.append(np.array([], dtype="|S8"))
                sort_idx_list.append(np.array([], dtype=int))
                host_mask_sorted_list.append(np.array([], dtype=bool))
                continue

            host_grp = f["HostTable"][transient_type]
            host_grps.append(host_grp)
            host_gids = host_grp["GID"][...]
            sort_idx = np.argsort(host_gids)
            gids_sorted = host_gids[sort_idx]

            host_mask = self._host_pass_mask(host_grp)
            host_mask_sorted = host_mask[sort_idx]

            gids_sorted_list.append(gids_sorted)
            sort_idx_list.append(sort_idx)
            host_mask_sorted_list.append(host_mask_sorted)

        return host_grps, gids_sorted_list, sort_idx_list, host_mask_sorted_list

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
        for b, mmax in zip(self.bands_to_filter, self.band_maxes):
            ds = subgrp[f"mag_{b}"]  # shape (N, T)
            # chunk along rows
            for i in range(0, N, batch):
                sl = slice(i, min(i + batch, N))
                arr = ds[sl]  # (B,T)
                # nanmin across time; if all NaN, result is NaN (treated as fail)
                new_arr = np.where(np.isnan(arr), np.inf, arr)
                ok = np.any(new_arr <= mmax, axis=1)
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

    def _build_subtype_shards(
        self,
        transient_type: str,
        subname: str,
        host_grps: list,
        gids_sorted_list: list,
        host_mask_sorted_list: list,
    ) -> tuple[list[_SubclassShard], int, int]:
        shards: list[_SubclassShard] = []
        total_rows = 0
        total_ok = 0

        # collect shards from each file
        for f_idx, f in enumerate(self.files):
            # skip if class/subclass missing in this file
            has_transient_type = transient_type in f["TransientTable"]
            if not has_transient_type:
                continue

            grp = f["TransientTable"][transient_type]
            has_subname = subname in grp
            if not has_subname:
                continue

            subgrp = grp[subname]
            no_host_table = host_grps[f_idx] is None
            if no_host_table:
                continue

            eligible_mask = self._transient_pass_mask(
                subgrp,
                gids_sorted_list[f_idx],
                host_mask_sorted_list[f_idx],
            )
            n_ok = int(eligible_mask.sum())
            if n_ok == 0:
                continue

            N = eligible_mask.size
            redshifts = subgrp["z"][:]
            if "AGN" in subname:
                weights = np.ones_like(redshifts) / len(redshifts)
            else:
                try:
                    rate_func = RATE_FUNCS[subname]
                except KeyError:
                    raise KeyError(
                        f"Transient subclass {subname} not found in rate functions."
                    )
                weights = rate_func(redshifts).astype(np.float64)
                weights[weights < 0] = 0.0

            if n_ok == N:
                eligible_idx = N
                weights_ok = weights
            else:
                eligible_idx = np.flatnonzero(eligible_mask).astype(np.int64)
                weights_ok = weights[eligible_idx]
            weight_sum = np.sum(weights_ok)
            normed_weights = weights_ok / weight_sum

            shards.append(
                _SubclassShard(
                    file_index=f_idx,
                    grp=subgrp,
                    N=N,
                    n_ok=n_ok,
                    eligible=eligible_idx,
                    weight_sum=weight_sum,
                    weights=normed_weights,
                )
            )
            total_rows += N
            total_ok += n_ok

        return shards, total_rows, total_ok

    def _get_expected_number(self, subname: str, total_ok: int) -> int:
        if subname in RATE_FUNCS:
            rate_fn = RATE_FUNCS[subname]
            n_expected = int(
                expected_number(
                    rate_fn=rate_fn,
                    cosmo=self._cosmo,
                    z_min=self.zmin,
                    z_max=self.zmax,
                )
            )
        elif "AGN" in subname:
            n_expected = total_ok
        else:
            raise KeyError(
                f"Transient Subclass {subname} not found in rate functions. "
                + f"Rate functions are available for {list(RATE_FUNCS.keys())}."
            )

        return n_expected

    def _build_subtype_indeces(
        self,
        transient_type: str,
        host_grps: list,
        gids_sorted_list: list,
        host_mask_sorted_list: list,
    ) -> tuple[
        list[_SubclassIndex],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        sub_list: list[_SubclassIndex] = []
        subclass_total = []
        subclass_selected = []
        subclass_expected = []

        for subname in self.transient_subtypes[transient_type]:

            shards, total_rows, total_ok = self._build_subtype_shards(
                transient_type=transient_type,
                subname=subname,
                host_grps=host_grps,
                gids_sorted_list=gids_sorted_list,
                host_mask_sorted_list=host_mask_sorted_list,
            )
            # keep only if any shard has survivors
            if not shards:
                continue

            # expected number for this subclass (same across files)
            n_expected = self._get_expected_number(subname=subname, total_ok=total_ok)

            sub_list.append(
                _SubclassIndex(name=subname, shards=shards, n_expected=n_expected)
            )
            subclass_total.append(total_rows)
            subclass_selected.append(total_ok)
            subclass_expected.append(n_expected)

        # sort subclasses by name for determinism
        sub_names = [s.name for s in sub_list]
        idx_ordered = np.argsort(sub_names)
        ordered_sub_list = [sub_list[i] for i in idx_ordered]
        subclass_total = np.asarray(subclass_total)[idx_ordered]
        subclass_selected = np.asarray(subclass_selected)[idx_ordered]
        subclass_expected = np.asarray(subclass_expected)[idx_ordered]

        return (ordered_sub_list, subclass_total, subclass_selected, subclass_expected)

    def _build_transient_index(self, transient_type: str) -> _ClassIndex:

        host_grps, gids_sorted_list, sort_idx_list, host_mask_sorted_list = (
            self._build_index_host_info(transient_type)
        )

        # Subclasses across files (as shards)

        ordered_sub_list, subclass_total, subclass_selected, subclass_expected = (
            self._build_subtype_indeces(
                transient_type=transient_type,
                host_grps=host_grps,
                gids_sorted_list=gids_sorted_list,
                host_mask_sorted_list=host_mask_sorted_list,
            )
        )

        total = int(np.sum(subclass_total))
        total_selected = int(np.sum(subclass_selected))
        total_expected = int(np.sum(subclass_expected))

        class_index = _ClassIndex(
            host_grp=host_grps,
            host_gid_sorted=gids_sorted_list,
            host_gid_sort_idx=sort_idx_list,
            host_mask_sorted=host_mask_sorted_list,
            subclasses=ordered_sub_list,
            subclass_total=subclass_total,
            subclass_selected=subclass_selected,
            subclass_expected=subclass_expected,
            subclass_weights=subclass_expected,  # placeholder; overwritten below
            total=total,
            total_expected=total_expected,
            total_selected=total_selected,
        )

        return class_index

    # -------------------- sampling --------------------

    def _sample_from_class(
        self, cls: str
    ) -> tuple[_SubclassIndex, _SubclassShard, int]:
        """Sample a transient subclass, a subclass shard and an index within
        that subclass shard over all surviving subclasses within the provided
        class.

        Parameters
        ----------
        cls : str
            Transient class name.

        Returns
        -------
        s : _SubclassIndex
            The sampled transient subclass.
        sh: _SubclassShard:
            The sampled subclass shard
        i : int
            Index within the subclass in the file
            belonging to the sampled shard.
        """

        ci = self._index[cls]

        # Ensure subclass weights are normalized (E_{rl} / sum E)
        p_sub = ci.subclass_weights
        p_sub = p_sub / p_sub.sum()

        s = ci.subclasses[self.rng.choice(len(ci.subclasses), p=p_sub)]

        # P(file | leaf) ∝ S_{f,rl} = shard.w_sum
        shard_weights = np.array([sh.weight_sum for sh in s.shards], dtype=float)
        shard_weights /= shard_weights.sum()
        sh = s.shards[self.rng.choice(len(s.shards), p=shard_weights)]

        # P(row | file, leaf) ∝ d_{rl}(z_i):
        i = int(self.rng.choice(sh.eligible, p=sh.weights))

        return s, sh, i

    def _host_lookup(self, cls: str, file_index: int, gid_bytes: bytes) -> int:
        """Given a transient class, a file index and a GID (as bytes), return
        the index of the corresponding host in the HostTable for that class.

        Parameters
        ----------
        cls: str
            Transient class name.
        file_index: int
            Sampled file index
        gid_bytes: bytes
            GID of the host as bytes (|S8).

        Returns
        -------
        int
            Index of the host in the HostTable for the given class.
        """
        ci = self._index[cls]
        gids_sorted = ci.host_gid_sorted[file_index]
        sort_idx = ci.host_gid_sort_idx[file_index]

        pos = int(np.searchsorted(gids_sorted, gid_bytes))
        if pos >= len(gids_sorted) or gids_sorted[pos] != gid_bytes:
            raise KeyError(f"GID {gid_bytes!r} not found in HostTable/{cls}")
        return int(sort_idx[pos])

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
        cls = self.rng.choice(self.active_transient_types, p=self.class_weights)
        s, sh, i = self._sample_from_class(cls)
        file_index = sh.file_index
        g = sh.grp

        transient_metadata = {
            "name": f"{s.name}",
            "z": float(g["z"][i]),
            "ra_off": float(g["ra_off"][i]),
            "dec_off": float(g["dec_off"][i]),
        }

        mjd = g["MJD"][i]
        transient_lightcurve = {}
        min_mag = np.inf
        for band in BANDS:

            mags = g[f"mag_{band}"][i]
            mags = np.where(mags == 99.0, np.inf, mags)

            idx_min_i = np.nanargmin(mags)
            min_mag_i = mags[idx_min_i]
            if min_mag_i < min_mag:
                min_mag = min_mag_i
                idx_min = idx_min_i

            transient_lightcurve[f"ps_mag_{band}"] = mags
        mjd = mjd - mjd[idx_min]
        transient_lightcurve["MJD"] = mjd

        transient_dict = transient_metadata | transient_lightcurve

        gid_b = g["GID"][i]
        host_idx = self._host_lookup(cls=cls, file_index=file_index, gid_bytes=gid_b)
        host_grp = self._index[cls].host_grp[file_index]
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
        for f in getattr(self, "files", []):
            try:
                f.close()
            except Exception:
                pass
