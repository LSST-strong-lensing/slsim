"""Assign quasars to plausible hosts, conditioned on quasar luminosity.

Both the black-hole mass relations and the Eddington-ratio distribution
are Gaussian in log space. Their convolution gives the host
probabilities, and their product gives the conditional Eddington-ratio
distribution after a host is drawn. See the accompanying README for the
model and references.
"""

import warnings

import numpy as np
from astropy.table import hstack
from tqdm import tqdm

# Eddington luminosity per solar mass [erg s^-1], 4 pi G M_sun m_p c / sigma_T
L_EDDINGTON_PER_MSUN = 1.2570e38

# Fitted to the Hbeta/Mg II sample of Wu & Shen (2022); see the README.
L3000_ZERO_POINT = 35.27

# Runnoe et al. (2012), table 1 and erratum.
RUNNOE12_ZETA_3000 = 5.18

# Gaussian distribution of log10(lambda), fitted to Wu & Shen (2022).
ERDF_LOCATION = -1.15
ERDF_WIDTH = 0.30

# Black hole mass relations, each of the form
#     log10(M_BH / Msun) = intercept + slope * log10(property / pivot)
# with a lognormal intrinsic scatter of ``scatter`` dex. The relation used for a
# galaxy is selected by its "galaxy_type".
BLACK_HOLE_MASS_RELATIONS = {
    # Kormendy & Ho (2013), equation 7.
    "red": {
        "property": "vel_disp",
        "pivot": 200.0,
        "intercept": 9 + np.log10(0.309),
        "slope": 4.38,
        "scatter": 0.29,
    },
    # Reines & Volonteri (2015), equation 5.
    "blue": {
        "property": "stellar_mass",
        "pivot": 1e11,
        "intercept": 7.45,
        "slope": 1.05,
        "scatter": 0.24,
    },
}


def _as_string_array(values):
    """Galaxy types as text, decoding the bytes a FITS file round-trips them
    to."""
    values = np.atleast_1d(np.asarray(values))
    if values.dtype.kind == "S":
        return np.char.decode(values)
    return values


def black_hole_mass(galaxy_type, vel_disp=None, stellar_mass=None, relations=None):
    """Mean black hole mass of each galaxy and the scatter about it.

    The relation is chosen per galaxy from its type, so that each is used only
    where it is calibrated: an M-sigma relation for bulge-dominated galaxies and
    an M_BH-M_star relation for disc-dominated ones. See
    :data:`BLACK_HOLE_MASS_RELATIONS` for the relations and their references.

    Only the properties the relations actually need have to be supplied.

    :param galaxy_type: type of each galaxy, a key of ``relations``
    :type galaxy_type: array_like of str
    :param vel_disp: velocity dispersion of each galaxy [km/s]
    :type vel_disp: array_like or None
    :param stellar_mass: total stellar mass of each galaxy [solar masses]
    :type stellar_mass: array_like or None
    :param relations: black hole mass relations, defaulting to
        :data:`BLACK_HOLE_MASS_RELATIONS`
    :type relations: dict or None
    :return: mean black hole mass [solar masses], intrinsic scatter [dex]
    :rtype: tuple of numpy.ndarray
    :raises ValueError: if a type is unknown or its property is missing
    """
    relations = BLACK_HOLE_MASS_RELATIONS if relations is None else relations
    types = _as_string_array(galaxy_type)
    properties = {"vel_disp": vel_disp, "stellar_mass": stellar_mass}

    unknown = set(np.unique(types)) - set(relations)
    if unknown:
        raise ValueError(
            "No black hole mass relation for galaxy type(s) %s. The known types "
            "are %s." % (sorted(unknown), list(relations))
        )

    log_mass = np.empty(types.shape)
    scatter = np.empty(types.shape)
    for name, relation in relations.items():
        selected = types == name
        if not selected.any():
            continue
        values = properties[relation["property"]]
        if values is None:
            raise ValueError(
                "The black hole mass relation of the '%s' galaxies needs a "
                "'%s' column." % (name, relation["property"])
            )
        values = np.asarray(values, dtype=float)[selected]
        # a non-positive property gives -inf here, which the caller drops
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mass[selected] = relation["intercept"] + relation["slope"] * np.log10(
                values / relation["pivot"]
            )
        scatter[selected] = relation["scatter"]

    return 10**log_mass, scatter


def log_bolometric_luminosity(m_i, scatter=0.0, rng=None):
    """Log10 bolometric luminosity for absolute magnitude ``M_i(z=2)``.

    ``m_i`` is the K-corrected absolute i-band magnitude normalised to z = 2,
    M_i(z=2), which is the quantity the Richards et al. (2006) / Oguri &
    Marshall (2010) luminosity function is written in. It fixes lambda
    L_lambda(3000 A) through :data:`L3000_ZERO_POINT`, and the bolometric
    correction :data:`RUNNOE12_ZETA_3000` turns that into a bolometric
    luminosity.

    :param m_i: absolute i-band magnitude M_i(z=2)
    :type m_i: float or numpy.ndarray
    :param scatter: object-to-object scatter of the bolometric correction [dex]
    :param rng: random number generator used for the scatter
    :type rng: numpy.random.Generator or None
    :return: log10 bolometric luminosity [erg/s]
    :rtype: float or numpy.ndarray
    """
    log_l3000 = L3000_ZERO_POINT - 0.4 * np.asarray(m_i, dtype=float)
    log_l_bol = log_l3000 + np.log10(RUNNOE12_ZETA_3000)
    if scatter:
        rng = np.random.default_rng() if rng is None else rng
        log_l_bol = log_l_bol + rng.normal(0, scatter, np.shape(log_l_bol))
    return log_l_bol


def bolometric_luminosity(m_i, scatter=0.0, rng=None):
    """Bolometric luminosity for absolute magnitude ``M_i(z=2)`` [erg/s]."""
    return 10 ** log_bolometric_luminosity(m_i, scatter=scatter, rng=rng)


def _draw_from_log_weights(log_weight, rng):
    """Draw an index from unnormalised log weights."""
    log_weight = np.asarray(log_weight, dtype=float)
    finite = np.isfinite(log_weight)
    if not finite.any():
        raise ValueError("Cannot draw from weights with no finite value.")
    weight = np.zeros_like(log_weight)
    weight[finite] = np.exp(log_weight[finite] - np.max(log_weight[finite]))
    cumulative = np.cumsum(weight)
    return int(np.searchsorted(cumulative, rng.random() * cumulative[-1]))


def _host_log_weights(c, mean_log_mass, mass_scatter, location, width):
    """Marginal log probability of each host for luminosity coordinate
    ``c``."""
    variance = mass_scatter**2 + width**2
    residual = c - mean_log_mass - location
    return -0.5 * (np.log(variance) + residual**2 / variance)


def _conditional_eddington_parameters(c, mean_log_mass, mass_scatter, location, width):
    """Mean and variance of log10(Eddington ratio), conditional on a host."""
    variance = 1.0 / (1.0 / width**2 + 1.0 / mass_scatter**2)
    mean = variance * (location / width**2 + (c - mean_log_mass) / mass_scatter**2)
    return mean, variance


class QuasarHostMatch(object):
    """Assign host galaxies, black hole masses and Eddington ratios to quasars.

    Host candidates are galaxies in a thin redshift slice around the
    quasar. A uniform prior over candidates therefore weights by galaxy
    number density.
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
        """
        self.quasar_catalog = quasar_catalog
        self.galaxy_catalog = galaxy_catalog.copy()
        self._delta_z = delta_z
        self._min_candidates = min_candidates
        self._max_delta_z = max_delta_z
        self._bc_scatter = bolometric_correction_scatter
        self._erdf_location = eddington_ratio_location
        self._erdf_width = eddington_ratio_width
        self._max_offset_sigma = max_offset_sigma
        self._rng = np.random.default_rng() if rng is None else rng

        # indices of the quasars no host galaxy could account for
        self.rejected_indices = []

    @property
    def n_rejected(self):
        """Number of quasars that could not be assigned a host galaxy."""
        return len(self.rejected_indices)

    def match(self, progress=True):
        """Match every quasar with a host galaxy.

        :param progress: whether to show a progress bar
        :return: catalog of the quasars that could be matched, joined
            with their host galaxies and with
            "black_hole_mass_exponent", "eddington_ratio" and
            "log_bolometric_luminosity" columns added
        :rtype: astropy Table
        """
        self._validate()
        galaxy_z, log_mass, scatter = self._prepare_galaxies()

        log_l_bol = log_bolometric_luminosity(
            np.asarray(self.quasar_catalog["M_i"], dtype=float),
            scatter=self._bc_scatter,
            rng=self._rng,
        )
        log_l_edd_per_msun = np.log10(L_EDDINGTON_PER_MSUN)

        quasar_z = np.asarray(self.quasar_catalog["z"], dtype=float)
        self.rejected_indices = []
        matched_quasars, matched_galaxies = [], []
        matched_log_mass, matched_eddington_ratio = [], []

        rows = tqdm(
            range(len(self.quasar_catalog)),
            desc="Matching quasars with host galaxies",
            disable=not progress,
        )
        for i in rows:
            start, end = self._candidate_range(galaxy_z, quasar_z[i])
            if start == end:
                self.rejected_indices.append(i)
                continue
            mean = log_mass[start:end]
            spread = scatter[start:end]
            c = log_l_bol[i] - log_l_edd_per_msun
            combined_scatter = np.sqrt(spread**2 + self._erdf_width**2)
            offset = (c - mean - self._erdf_location) / combined_scatter
            if np.min(np.abs(offset)) > self._max_offset_sigma:
                self.rejected_indices.append(i)
                continue

            host = _draw_from_log_weights(
                _host_log_weights(
                    c,
                    mean,
                    spread,
                    self._erdf_location,
                    self._erdf_width,
                ),
                self._rng,
            )
            conditional_mean, conditional_variance = _conditional_eddington_parameters(
                c,
                mean[host],
                spread[host],
                self._erdf_location,
                self._erdf_width,
            )
            log_lambda = self._rng.normal(
                conditional_mean, np.sqrt(conditional_variance)
            )
            log_mass_bh = c - log_lambda

            matched_quasars.append(i)
            matched_galaxies.append(start + host)
            matched_log_mass.append(log_mass_bh)
            matched_eddington_ratio.append(10**log_lambda)

        self._warn_about_rejections()

        quasars = self.quasar_catalog[matched_quasars]
        galaxies = self.galaxy_catalog[matched_galaxies]
        galaxies["black_hole_mass_exponent"] = matched_log_mass
        galaxies["eddington_ratio"] = matched_eddington_ratio
        galaxies["log_bolometric_luminosity"] = log_l_bol[matched_quasars]
        galaxies.remove_column("z")

        return hstack(
            [quasars, galaxies],
            table_names=["quasar", "host_galaxy"],
            join_type="exact",
        )

    def _validate(self):
        for column in ("z", "M_i"):
            if column not in self.quasar_catalog.colnames:
                raise ValueError(
                    "The quasar catalog needs a '%s' column to perform the "
                    "quasar-host match." % column
                )
        for column in ("z", "galaxy_type"):
            if column not in self.galaxy_catalog.colnames:
                raise ValueError(
                    "The galaxy catalog needs a '%s' column to perform the "
                    "quasar-host match." % column
                )
        if self._erdf_width <= 0:
            raise ValueError("eddington_ratio_width must be positive.")

    def _prepare_galaxies(self):
        """Redshift, log10 mean black hole mass and scatter, sorted by
        redshift.

        Galaxies whose black hole mass relation cannot be evaluated,
        because the property it uses is missing or non-positive, carry
        no information about the black hole and are dropped.
        """
        properties = {
            name: self.galaxy_catalog[name]
            for name in ("vel_disp", "stellar_mass")
            if name in self.galaxy_catalog.colnames
        }
        mass, scatter = black_hole_mass(
            self.galaxy_catalog["galaxy_type"], **properties
        )

        usable = np.isfinite(mass) & (mass > 0)
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
        mass = mass[order]
        return redshift[order], np.log10(mass), scatter[order]

    def _candidate_range(self, galaxy_z, redshift):
        """Bounds of the redshift slice the host candidates are drawn from,
        widened geometrically until it holds ``min_candidates`` galaxies."""
        delta_z = self._delta_z
        while True:
            start = np.searchsorted(galaxy_z, redshift - delta_z, side="left")
            end = np.searchsorted(galaxy_z, redshift + delta_z, side="right")
            if end - start >= self._min_candidates or delta_z >= self._max_delta_z:
                return start, end
            delta_z = min(delta_z * 2, self._max_delta_z)

    def _warn_about_rejections(self):
        """Report the quasars no host galaxy could account for, and where in
        magnitude and redshift they sit."""
        if not self.rejected_indices:
            return

        def span(column):
            values = np.asarray(self.quasar_catalog[column], dtype=float)[
                self.rejected_indices
            ]
            return "%.2f to %.2f (median %.2f)" % (
                values.min(),
                values.max(),
                np.median(values),
            )

        warnings.warn(
            "%d of %d quasars (%.1f%%) were dropped: no candidate host is "
            "within %g combined standard deviations of the luminosity "
            "implied by its mass relation and the Eddington-ratio distribution. "
            "They span M_i %s and z %s."
            % (
                self.n_rejected,
                len(self.quasar_catalog),
                100 * self.n_rejected / len(self.quasar_catalog),
                self._max_offset_sigma,
                span("M_i"),
                span("z"),
            ),
            UserWarning,
        )
