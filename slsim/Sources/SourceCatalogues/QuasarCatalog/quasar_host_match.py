"""Assign quasars to host galaxies, conditioned on quasar luminosity.

The black-hole mass relations and the Eddington-ratio distribution are both
Gaussian in log space, so the host probabilities and the conditional
Eddington ratio have closed forms. See the README for the model.
"""

import warnings

import numpy as np
from astropy.table import hstack
from tqdm import tqdm

# Eddington luminosity per solar mass [erg s^-1], 4 pi G M_sun m_p c / sigma_T
L_EDDINGTON_PER_MSUN = 1.2570e38

# log10 lambda L_lambda(3000 A) = L3000_ZERO_POINT - 0.4 M_i(z=2), fitted to the
# Hbeta/Mg II sample of Wu & Shen (2022). A pure alpha_nu = -0.5 continuum in the
# Richards et al. (2006) M_i(z=2) convention gives 35.20; the data sit 0.07 dex above
L3000_ZERO_POINT = 35.27

# L_bol = 5.15 lambda L_lambda(3000 A): Richards et al. (2006), as used by Shen et al.
# (2011) and Wu & Shen (2022). Runnoe et al. (2012) give 5.2 +- 0.2
BOLOMETRIC_CORRECTION_3000 = 5.15

# Gaussian distribution of log10(Eddington ratio), fitted to Wu & Shen (2022)
ERDF_LOCATION = -1.15
ERDF_WIDTH = 0.30

# log10(M_BH / Msun) = intercept + slope * log10(property / pivot), with a lognormal
# intrinsic scatter [dex]; the relation is chosen per galaxy by its "galaxy_type"
BLACK_HOLE_MASS_RELATIONS = {
    # Kormendy & Ho (2013), equation 7
    "red": {
        "property": "vel_disp",
        "pivot": 200.0,
        "intercept": 9 + np.log10(0.309),
        "slope": 4.38,
        "scatter": 0.29,
    },
    # Reines & Volonteri (2015), equation 5
    "blue": {
        "property": "stellar_mass",
        "pivot": 1e11,
        "intercept": 7.45,
        "slope": 1.05,
        "scatter": 0.24,
    },
}


def log_black_hole_mass(galaxy_type, vel_disp=None, stellar_mass=None):
    """Mean log10 black hole mass of each galaxy and the intrinsic scatter about
    it, using the relation of its type (see :data:`BLACK_HOLE_MASS_RELATIONS`).

    :param galaxy_type: type of each galaxy, a key of the relations
    :param vel_disp: velocity dispersion of each galaxy [km/s]
    :param stellar_mass: total stellar mass of each galaxy [solar masses]
    :return: log10 mean black hole mass [solar masses], scatter [dex]
    :rtype: tuple of numpy.ndarray
    """
    types = np.atleast_1d(np.asarray(galaxy_type)).astype(str)
    properties = {"vel_disp": vel_disp, "stellar_mass": stellar_mass}

    unknown = set(np.unique(types)) - set(BLACK_HOLE_MASS_RELATIONS)
    if unknown:
        raise ValueError(
            f"No black hole mass relation for galaxy type(s) {sorted(unknown)}."
        )

    log_mass = np.empty(types.shape)
    scatter = np.empty(types.shape)
    for name, relation in BLACK_HOLE_MASS_RELATIONS.items():
        selected = types == name
        if not selected.any():
            continue
        values = properties[relation["property"]]
        if values is None:
            raise ValueError(
                f"The '{name}' galaxies need a '{relation['property']}' column."
            )
        values = np.asarray(values, dtype=float)[selected]
        # a non-positive property gives -inf/nan here, which the matcher drops
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mass[selected] = relation["intercept"] + relation["slope"] * np.log10(
                values / relation["pivot"]
            )
        scatter[selected] = relation["scatter"]
    return log_mass, scatter


def log_bolometric_luminosity(m_i, scatter=0.0, rng=None):
    """Log10 bolometric luminosity [erg/s] for absolute magnitude M_i(z=2).

    :param m_i: absolute i-band magnitude M_i(z=2)
    :param scatter: object-to-object scatter of the bolometric correction [dex]
    :param rng: random number generator used for the scatter
    :type rng: numpy.random.Generator or None
    """
    log_l_bol = (
        L3000_ZERO_POINT
        - 0.4 * np.asarray(m_i, dtype=float)
        + np.log10(BOLOMETRIC_CORRECTION_3000)
    )
    if scatter:
        rng = np.random.default_rng() if rng is None else rng
        log_l_bol = log_l_bol + rng.normal(0, scatter, np.shape(log_l_bol))
    return log_l_bol


class QuasarHostMatch:
    """Assign host galaxies, black hole masses and Eddington ratios to quasars.

    Host candidates are the galaxies in a thin redshift slice around the
    quasar, so a uniform prior over candidates weights by galaxy number density.
    """

    def __init__(
        self,
        quasar_catalog,
        galaxy_catalog,
        delta_z=0.001,
        min_candidates=50,
        max_delta_z=0.05,
        bolometric_correction_scatter=0.1,
        eddington_ratio_location=ERDF_LOCATION,
        eddington_ratio_width=ERDF_WIDTH,
        max_offset_sigma=4.0,
        rng=None,
        progress=True,
    ):
        """
        :param quasar_catalog: quasar catalog with a redshift column "z" and an
            absolute i-band magnitude column "M_i" in the M_i(z=2) system
        :type quasar_catalog: astropy Table
        :param galaxy_catalog: host galaxy candidates, with a "z" column, a
            "galaxy_type" column, and whichever of "vel_disp" and
            "stellar_mass" the relations of those types need
        :type galaxy_catalog: astropy Table
        :param delta_z: half-width of the redshift slice host candidates are
            drawn from
        :param min_candidates: the slice is widened until it holds at least this
            many galaxies, up to ``max_delta_z``
        :param max_delta_z: largest half-width the slice may be widened to
        :param bolometric_correction_scatter: object-to-object scatter of the
            bolometric correction [dex]
        :param eddington_ratio_location: mean of log10(Eddington ratio)
        :param eddington_ratio_width: standard deviation of log10(Eddington
            ratio) [dex]
        :param max_offset_sigma: a quasar is rejected if no candidate host can
            produce it within this many combined standard deviations of the
            mass relation and Eddington-ratio distribution
        :param rng: random number generator, for reproducible catalogs
        :type rng: numpy.random.Generator or None
        :param progress: whether to show a progress bar
        """
        self.quasar_catalog = quasar_catalog
        self.galaxy_catalog = galaxy_catalog
        self.delta_z = delta_z
        self.min_candidates = min_candidates
        self.max_delta_z = max_delta_z
        self.bc_scatter = bolometric_correction_scatter
        self.erdf_location = eddington_ratio_location
        self.erdf_width = eddington_ratio_width
        self.max_offset_sigma = max_offset_sigma
        self.rng = np.random.default_rng() if rng is None else rng
        self.progress = progress
        # indices of the quasars no host galaxy could account for
        self.rejected_indices = []

    def match(self):
        """Match every quasar with a host galaxy.

        :return: the quasars that could be matched, joined with their host
            galaxies and with "black_hole_mass_exponent", "eddington_ratio" and
            "log_bolometric_luminosity" columns added
        :rtype: astropy Table
        """
        self._validate()
        galaxy_z, log_mass, scatter = self._prepare_galaxies()
        quasar_z = np.asarray(self.quasar_catalog["z"], dtype=float)
        log_l_bol = log_bolometric_luminosity(
            self.quasar_catalog["M_i"], scatter=self.bc_scatter, rng=self.rng
        )
        # log10 of the black hole mass at an Eddington ratio of one
        c_all = log_l_bol - np.log10(L_EDDINGTON_PER_MSUN)

        self.rejected_indices = []
        matched = []  # (quasar index, galaxy index, log M_BH, log lambda)
        for i in tqdm(
            range(len(quasar_z)),
            desc="Matching quasars with host galaxies",
            disable=not self.progress,
        ):
            start, end = self._candidate_range(galaxy_z, quasar_z[i])
            mean, spread, c = log_mass[start:end], scatter[start:end], c_all[i]
            variance = spread**2 + self.erdf_width**2
            offset = (c - mean - self.erdf_location) / np.sqrt(variance)
            if start == end or np.abs(offset).min() > self.max_offset_sigma:
                self.rejected_indices.append(i)
                continue

            log_weight = -0.5 * (np.log(variance) + offset**2)
            weight = np.exp(log_weight - log_weight.max())
            host = self.rng.choice(len(weight), p=weight / weight.sum())

            # Eddington ratio conditional on the host: product of two Gaussians
            v = 1 / (1 / self.erdf_width**2 + 1 / spread[host] ** 2)
            x = self.rng.normal(
                v
                * (
                    self.erdf_location / self.erdf_width**2
                    + (c - mean[host]) / spread[host] ** 2
                ),
                np.sqrt(v),
            )
            matched.append((i, start + host, c - x, x))

        self._warn_about_rejections()
        matched = np.array(matched).reshape(-1, 4)
        quasar_index, galaxy_index = matched[:, :2].astype(int).T
        quasars = self.quasar_catalog[quasar_index]
        galaxies = self.galaxy_catalog[galaxy_index]
        galaxies["black_hole_mass_exponent"] = matched[:, 2]
        galaxies["eddington_ratio"] = 10 ** matched[:, 3]
        galaxies["log_bolometric_luminosity"] = log_l_bol[quasar_index]
        galaxies.remove_column("z")
        return hstack(
            [quasars, galaxies],
            table_names=["quasar", "host_galaxy"],
            join_type="exact",
        )

    def _validate(self):
        for table, name, columns in (
            (self.quasar_catalog, "quasar", ("z", "M_i")),
            (self.galaxy_catalog, "galaxy", ("z", "galaxy_type")),
        ):
            for column in columns:
                if column not in table.colnames:
                    raise ValueError(f"The {name} catalog needs a '{column}' column.")
        if self.erdf_width <= 0:
            raise ValueError("eddington_ratio_width must be positive.")

    def _prepare_galaxies(self):
        """Sort the galaxies by redshift, dropping those whose relation cannot
        be evaluated, and return their redshift, log mass and scatter."""
        properties = {
            name: self.galaxy_catalog[name]
            for name in ("vel_disp", "stellar_mass")
            if name in self.galaxy_catalog.colnames
        }
        log_mass, scatter = log_black_hole_mass(
            self.galaxy_catalog["galaxy_type"], **properties
        )
        usable = np.isfinite(log_mass)
        if not usable.any():
            raise ValueError(
                "No galaxy in the catalog has a usable black hole mass. Check "
                "that the properties the relations of these galaxy types need "
                "are present and positive."
            )
        # dtype=float forces native byte order: a catalog read from FITS is
        # big-endian, and numpy has no fast searchsorted path for that, which
        # costs ~25 ms per lookup instead of ~1 us
        redshift = np.asarray(self.galaxy_catalog["z"], dtype=float)
        order = np.flatnonzero(usable)[np.argsort(redshift[usable])]
        self.galaxy_catalog = self.galaxy_catalog[order]
        return redshift[order], log_mass[order], scatter[order]

    def _candidate_range(self, galaxy_z, redshift):
        """Bounds of the redshift slice, widened geometrically until it holds
        ``min_candidates`` galaxies."""
        delta_z = self.delta_z
        while True:
            start = np.searchsorted(galaxy_z, redshift - delta_z, side="left")
            end = np.searchsorted(galaxy_z, redshift + delta_z, side="right")
            if end - start >= self.min_candidates or delta_z >= self.max_delta_z:
                return start, end
            delta_z = min(delta_z * 2, self.max_delta_z)

    def _warn_about_rejections(self):
        if not self.rejected_indices:
            return
        n, n_total = len(self.rejected_indices), len(self.quasar_catalog)
        rejected = self.quasar_catalog[self.rejected_indices]
        m_i, z = (
            np.percentile(np.asarray(rejected[column], dtype=float), [0, 100, 50])
            for column in ("M_i", "z")
        )
        warnings.warn(
            f"{n} of {n_total} quasars ({100 * n / n_total:.1f}%) were dropped: no "
            f"candidate host is within {self.max_offset_sigma:g} combined standard "
            "deviations of the luminosity implied by its mass relation and the "
            f"Eddington-ratio distribution. They span M_i {m_i[0]:.2f} to {m_i[1]:.2f} "
            f"(median {m_i[2]:.2f}) and z {z[0]:.2f} to {z[1]:.2f} (median {z[2]:.2f}).",
            UserWarning,
        )
