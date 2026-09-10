"""Match quasars drawn from a luminosity function to plausible host galaxies.

The luminosity function already fixes how many quasars there are and how bright
they are, so the luminosity is an input here, not an output. What is left to
draw is the host galaxy, the black hole mass and the Eddington ratio, from their
joint distribution *conditioned on* that luminosity:

.. math::
    p(k, \\lambda) \\propto w_k\\, p(\\lambda)\\,
    \\mathcal{N}\\!\\left(\\log M_{BH}^{req}(\\lambda) \\,\\middle|\\,
    \\log M_{BH}(k),\\, s_k\\right)

:math:`w_k` is the prior over candidate hosts, the Gaussian is the intrinsic
scatter of galaxy ``k``'s black hole mass relation, and

.. math::
    M_{BH}^{req}(\\lambda) = \\frac{L_{bol}}{\\lambda\\, L_{Edd,1}}

is the mass that reproduces the luminosity at that Eddington ratio.
:math:`L_{Edd,1}` there is the Eddington luminosity *per solar mass*, the
constant :data:`L_EDDINGTON_PER_MSUN` — not the Eddington luminosity of the
black hole itself, which would make the expression circular. Because
:math:`L_{Edd}` is strictly linear in mass, :math:`L_{Edd}(M) = M L_{Edd,1}`,
the definition :math:`\\lambda \\equiv L_{bol}/L_{Edd}(M_{BH})` inverts to the
line above, and dividing an erg/s by an erg/s per solar mass leaves a mass.

The Gaussian is what conditioning on the luminosity looks like: a galaxy does
not have a black hole mass, it has a distribution of them, and that
distribution is the only thing tying a host to a quasar.

:meth:`QuasarHostMatch.match` samples it as two one-dimensional draws:

1. :math:`\\lambda` from :math:`p(\\lambda)\\,\\Phi(\\log M_{BH}^{req}(\\lambda))`,
   where :math:`\\Phi` is the candidate hosts' black hole mass function smoothed
   by their scatter. Setting :math:`M_{BH} = M_{BH}^{req}(\\lambda)` then makes
   :math:`L_{bol} = \\lambda L_{Edd}` exact.
2. the host from :math:`w_k \\mathcal{N}(\\log M_{BH} | \\log M_{BH}(k), s_k)`.

Because the scatter is a weight rather than a cut, the assigned black hole
masses scatter about the mean relation instead of lying on it.

Ingredients: :data:`BLACK_HOLE_MASS_RELATIONS` for the mass relations,
:func:`eddington_ratio_grid` for :math:`p(\\lambda)`, and
:func:`bolometric_luminosity` for :math:`L_{bol}`.
"""

import warnings

import numpy as np
from astropy.table import hstack
from tqdm import tqdm

# Eddington luminosity per solar mass [erg s^-1], 4 pi G M_sun m_p c / sigma_T
L_EDDINGTON_PER_MSUN = 1.2570e38

# log10(lambda L_lambda(3000 A) / erg s^-1) = L3000_ZERO_POINT - 0.4 * M_i(z=2).
# The slope follows from both sides being log luminosities; the zero point is
# fitted to the Hbeta/Mg II subsample of Wu & Shen (2022), ApJS 263, 42, and
# holds to 0.02 dex over M_i = -28 to -23 and 0.05 dex over z = 0.7 to 2.5.
# Reading the flux off a model SED anchored to M_i instead runs 0.18 dex faint.
L3000_ZERO_POINT = 35.27

# Runnoe et al. (2012), MNRAS 422, 478, erratum MNRAS 427, 1800, table 1:
# L_bol = zeta * lambda L_lambda(3000 A). Wu & Shen use 5.15 from Richards
# et al. (2006), ApJS 166, 470, which is the same number to 0.003 dex.
RUNNOE12_ZETA_3000 = 5.18

# Eddington ratio distribution, lognormal in log10(lambda) as broad-line
# quasars are observed to be (Kelly & Shen 2013, ApJ 764, 45; Schulze et al.
# 2015, MNRAS 447, 2085). Both are fitted to Wu & Shen (2022) at fixed
# bolometric luminosity, where the survey flux limit barely bites. The width is
# the one whose intrinsic spread plus the catalog's own 0.12 dex mass errors
# reproduce the observed spread, rather than matching it outright. See the
# README for why this replaced a bounded power law in the specific accretion
# rate of galaxies, which counts the quasar selection twice.
ERDF_LOCATION = -1.15
ERDF_WIDTH = 0.30

# Half-width of the Eddington ratio grid, in units of ERDF_WIDTH
ERDF_GRID_HALF_WIDTH = 4.0

# Black hole mass at which an optional mass-dependent duty cycle is normalised
DUTY_CYCLE_PIVOT_MSUN = 1e8

# Black hole mass relations, each of the form
#     log10(M_BH / Msun) = intercept + slope * log10(property / pivot)
# with a lognormal intrinsic scatter of ``scatter`` dex. The relation used for a
# galaxy is selected by its "galaxy_type".
BLACK_HOLE_MASS_RELATIONS = {
    # Kormendy & Ho (2013), ARA&A 51, 511, equation 7: the M-sigma relation of
    # ellipticals and classical bulges, M_BH = 0.309e9 Msun at sigma = 200
    # km/s, with 0.29 dex of intrinsic scatter. It is calibrated on bulge
    # velocity dispersions, so it is applied only to bulge-dominated hosts.
    "red": {
        "property": "vel_disp",
        "pivot": 200.0,
        "intercept": 9 + np.log10(0.309),
        "slope": 4.38,
        "scatter": 0.29,
    },
    # Reines & Volonteri (2015), ApJ 813, 82, equation 5: the M_BH-M_star
    # relation of local broad-line AGN, whose hosts are mostly disc-dominated
    # and whose normalisation sits more than a dex below the early-type one.
    # A disc galaxy's dispersion is not a bulge dispersion, so mass is used.
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


def eddington_ratio_grid(n_grid=32, location=ERDF_LOCATION, width=ERDF_WIDTH):
    """Grid of Eddington ratios and the probability of each.

    The distribution is lognormal, ``dP/dlog10(lambda)`` a Gaussian of mean
    ``location`` and standard deviation ``width``. Its normalisation would set
    the fraction of galaxies that are active, but the abundance of quasars is
    already fixed by the input luminosity function, so only the shape is used.

    The grid spans :data:`ERDF_GRID_HALF_WIDTH` standard deviations either side
    of the mean and carries equal intervals in dex, so the weights are the
    per-dex density itself and a draw may be spread across its own cell.

    :param n_grid: number of grid points
    :param location: mean of log10(Eddington ratio)
    :param width: standard deviation of log10(Eddington ratio) [dex]
    :return: grid of log10(Eddington ratio), normalised weight of each point
    :rtype: tuple of numpy.ndarray
    """
    reach = ERDF_GRID_HALF_WIDTH * width
    log_grid = np.linspace(location - reach, location + reach, n_grid)
    weight = np.exp(-0.5 * ((log_grid - location) / width) ** 2)
    return log_grid, weight / weight.sum()


def bolometric_luminosity(m_i, scatter=0.0, rng=None):
    """Bolometric luminosity of a quasar of absolute i-band magnitude ``m_i``.

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
    :return: bolometric luminosity [erg/s]
    :rtype: float or numpy.ndarray
    """
    log_l3000 = L3000_ZERO_POINT - 0.4 * np.asarray(m_i, dtype=float)
    log_l_bol = log_l3000 + np.log10(RUNNOE12_ZETA_3000)
    if scatter:
        rng = np.random.default_rng() if rng is None else rng
        log_l_bol = log_l_bol + rng.normal(0, scatter, np.shape(log_l_bol))
    return 10**log_l_bol


class QuasarHostMatch(object):
    """Assign host galaxies, black hole masses and Eddington ratios to quasars.

    See the module docstring for the distribution being sampled. Host candidates
    are the galaxies in a thin redshift slice around the quasar, so a uniform
    prior over them weights by galaxy number density; ``duty_cycle_slope`` tilts
    that prior towards more massive black holes.
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
        n_eddington_grid=32,
        max_offset_sigma=4.0,
        duty_cycle_slope=0.0,
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
        :param n_eddington_grid: number of Eddington ratio grid points
        :param max_offset_sigma: a quasar is rejected if no candidate host can
            produce it within this many times the scatter of its black hole mass
            relation
        :param duty_cycle_slope: exponent of an optional mass-dependent duty
            cycle, which weights a candidate by ``(M_BH /
            DUTY_CYCLE_PIVOT_MSUN)**duty_cycle_slope``. The default of zero
            leaves the prior uniform over candidates, so that hosts are weighted
            purely by their number density. A positive value makes massive black
            holes more likely to be active, which is observed but is a
            calibration rather than a first-principles ingredient.
        :param rng: random number generator, for reproducible catalogs
        :type rng: numpy.random.Generator or None
        :param progress: whether to show a progress bar
        """
        self.quasar_catalog = quasar_catalog
        self.galaxy_catalog = galaxy_catalog.copy()
        self._delta_z = delta_z
        self._min_candidates = min_candidates
        self._max_delta_z = max_delta_z
        self._bc_scatter = bolometric_correction_scatter
        self._erdf_location = eddington_ratio_location
        self._erdf_width = eddington_ratio_width
        self._n_eddington_grid = n_eddington_grid
        self._max_offset_sigma = max_offset_sigma
        self._duty_cycle_slope = duty_cycle_slope
        self._rng = np.random.default_rng() if rng is None else rng
        self._progress = progress

        # indices of the quasars no host galaxy could account for
        self.rejected_indices = []

    @property
    def n_rejected(self):
        """Number of quasars that could not be assigned a host galaxy."""
        return len(self.rejected_indices)

    def match(self):
        """Match every quasar with a host galaxy.

        :return: catalog of the quasars that could be matched, joined
            with their host galaxies and with
            "black_hole_mass_exponent", "eddington_ratio" and
            "log_bolometric_luminosity" columns added
        :rtype: astropy Table
        """
        self._validate()
        galaxy_z, log_mass, scatter, weight = self._prepare_galaxies()

        log_l_bol = np.log10(
            bolometric_luminosity(
                np.asarray(self.quasar_catalog["M_i"], dtype=float),
                scatter=self._bc_scatter,
                rng=self._rng,
            )
        )
        log_lambda_grid, lambda_weight = eddington_ratio_grid(
            self._n_eddington_grid, self._erdf_location, self._erdf_width
        )
        half_cell = 0.5 * (log_lambda_grid[1] - log_lambda_grid[0])
        log_l_edd_per_msun = np.log10(L_EDDINGTON_PER_MSUN)

        quasar_z = np.asarray(self.quasar_catalog["z"], dtype=float)
        self.rejected_indices = []
        matched_quasars, matched_galaxies = [], []
        matched_log_mass, matched_eddington_ratio = [], []

        rows = tqdm(
            range(len(self.quasar_catalog)),
            desc="Matching quasars with host galaxies",
            disable=not self._progress,
        )
        for i in rows:
            start, end = self._candidate_range(galaxy_z, quasar_z[i])
            if start == end:
                self.rejected_indices.append(i)
                continue
            mean = log_mass[start:end]
            spread = scatter[start:end]
            prior = weight[start:end]

            # mass each grid Eddington ratio would require of this quasar, and
            # how many scatters each candidate sits from it
            required = log_l_bol[i] - log_l_edd_per_msun - log_lambda_grid
            offset = (required[None, :] - mean[:, None]) / spread[:, None]
            if np.abs(offset).min() > self._max_offset_sigma:
                self.rejected_indices.append(i)
                continue

            # step 1: the Eddington ratio, with the hosts summed out. That sum
            # is the candidates' black hole mass function smoothed by their
            # intrinsic scatter, evaluated at the required mass.
            smoothed_mass_function = (
                prior[:, None] * np.exp(-0.5 * offset**2) / spread[:, None]
            ).sum(axis=0)
            cell = self._draw(lambda_weight * smoothed_mass_function)
            log_lambda = log_lambda_grid[cell] + self._rng.uniform(
                -half_cell, half_cell
            )
            # the mass reproducing the luminosity at exactly this lambda, so
            # that L_bol = lambda * L_Edd holds for the reported values
            log_mass_bh = log_l_bol[i] - log_l_edd_per_msun - log_lambda

            # step 2: the host, from the same Gaussian at that fixed mass
            host_offset = (log_mass_bh - mean) / spread
            host_index = start + self._draw(
                prior * np.exp(-0.5 * host_offset**2) / spread
            )

            matched_quasars.append(i)
            matched_galaxies.append(host_index)
            matched_log_mass.append(log_mass_bh)
            matched_eddington_ratio.append(10**log_lambda)

        self._warn_about_rejections(log_lambda_grid)

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

    def _prepare_galaxies(self):
        """Redshift, log10 mean black hole mass, intrinsic scatter and duty
        cycle weight of each candidate, sorted by redshift.

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
        weight = (mass / DUTY_CYCLE_PIVOT_MSUN) ** self._duty_cycle_slope
        return redshift[order], np.log10(mass), scatter[order], weight

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

    def _draw(self, weight):
        """Index drawn with probability proportional to ``weight``."""
        cumulative = np.cumsum(weight)
        index = np.searchsorted(cumulative, self._rng.random() * cumulative[-1])
        return int(min(index, weight.size - 1))

    def _warn_about_rejections(self, log_lambda_grid):
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
            "%d of %d quasars (%.1f%%) were dropped: no candidate host has a "
            "black hole mass relation reaching their luminosity within %g sigma. "
            "They span M_i %s and z %s. The Eddington ratio grid runs over "
            "log10(lambda) %.2f to %.2f, which bounds the mass a quasar of a "
            "given luminosity can have."
            % (
                self.n_rejected,
                len(self.quasar_catalog),
                100 * self.n_rejected / len(self.quasar_catalog),
                self._max_offset_sigma,
                span("M_i"),
                span("z"),
                log_lambda_grid[0],
                log_lambda_grid[-1],
            ),
            UserWarning,
        )
