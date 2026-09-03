"""Match quasars drawn from a luminosity function to plausible host galaxies.

The quasar catalog fixes the abundance and luminosity of the AGN; this
module assigns each of them a host galaxy, a black hole mass and an
Eddington ratio that are jointly consistent with

* the M-sigma relation of Kormendy & Ho (2013) *including* its 0.29 dex
  intrinsic scatter,
* an Eddington ratio distribution, either the power law of Korytov et al.
  (2019) or a lognormal calibrated on the broad-line quasars of Shen et al.
  (2011),
* the bolometric correction of Runnoe et al. (2012), anchored to the same
  qsogen SED that generates the broad-band photometry.

The assignment is a weighted draw from the joint posterior rather than a
nearest-neighbour match, so the sampled Eddington ratios follow the input
distribution and the resulting (M_BH, sigma) pairs scatter around the
M-sigma relation instead of lying exactly on it.
"""

from functools import lru_cache
import warnings

import numpy as np
import speclite.filters
from astropy.cosmology import FlatLambdaCDM
from astropy.table import hstack
from scipy.interpolate import interp1d
from tqdm import tqdm

from slsim.Sources.SourceCatalogues.QuasarCatalog.qsogen import Quasar_sed, params_agile

# Eddington luminosity per solar mass [erg s^-1], 4 pi G M_sun m_p c / sigma_T
L_EDDINGTON_PER_MSUN = 1.2570e38

# Runnoe et al. (2012), erratum table 1: L_iso = zeta * lambda L_lambda and
# log L_iso = A + B log(lambda L_lambda), for lambda L_lambda in erg/s.
RUNNOE12_BOLOMETRIC_CORRECTION = {
    1450: {"zeta": 4.20, "A": 4.745, "B": 0.910},
    3000: {"zeta": 5.18, "A": 1.852, "B": 0.975},
    5100: {"zeta": 8.10, "A": 4.891, "B": 0.912},
}

# Reference monochromatic luminosity used to anchor the qsogen SED [log erg/s].
_LOG_L3000_REFERENCE = 46.0


def sample_eddington_rate(
    z,
    z0=0.6,
    gamma_e=-0.65,
    gamma_z=3.47,
    A=0.00071,
    lambda_min=0.1,
    lambda_max=1.0,
    size=1,
    n_grid=1000,
):
    """Sample Eddington ratios from the Korytov et al. (2019) distribution.

    The distribution of Eq. (16) of Korytov et al. (2019),
    https://arxiv.org/abs/1907.06530, is

    .. math::
        P(\\lambda|z) = A \\frac{1+z}{(1+z_0)^{\\gamma_z}} \\lambda^{\\gamma_e}

    Note that the redshift dependence lives entirely in the normalisation,
    where it sets the fraction of galaxies that are active. Once the
    distribution is normalised to draw from it, the shape is the pure power
    law ``lambda**gamma_e`` and the returned samples are independent of ``z``.
    That is the intended behaviour here, because the abundance of quasars is
    already fixed by the luminosity function of the input catalog. The ``z``,
    ``z0``, ``gamma_z`` and ``A`` arguments are kept for interface
    compatibility and are not used.

    :param z: redshift at which to sample Eddington ratios
    :type z: float
    :param z0: reference redshift of the amplitude evolution (unused)
    :param gamma_e: power-law index of the Eddington ratio distribution
    :param gamma_z: index of the amplitude redshift evolution (unused)
    :param A: amplitude of the distribution (unused)
    :param lambda_min: minimum Eddington ratio to sample
    :param lambda_max: maximum Eddington ratio to sample
    :param size: number of samples to generate
    :param n_grid: number of grid points for the numerical CDF
    :return: sampled Eddington ratios
    :rtype: numpy.ndarray
    """
    grid, pdf = eddington_ratio_grid(
        gamma_e=gamma_e,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        n_grid=n_grid,
    )
    cdf = np.concatenate([[0.0], np.cumsum(np.diff(grid) * 0.5 * (pdf[1:] + pdf[:-1]))])
    cdf /= cdf[-1]
    inv_cdf = interp1d(cdf, grid)
    return inv_cdf(np.random.uniform(0, 1, size))


# Default Eddington ratio bounds for each distribution shape.
DEFAULT_LAMBDA_BOUNDS = {"power_law": (0.01, 1.0), "lognormal": (1e-3, 3.0)}

# Lognormal Eddington ratio distribution measured from the broad-line quasars of
# Shen et al. (2011), https://arxiv.org/abs/1006.5178: mean and standard
# deviation of log10(lambda) over the whole DR7 sample.
SHEN11_LOG_LAMBDA_MEAN = -0.77
SHEN11_LOG_LAMBDA_SIGMA = 0.42


def eddington_ratio_grid(
    gamma_e=-0.65,
    lambda_min=0.01,
    lambda_max=1.0,
    n_grid=1000,
    distribution="power_law",
    log_mean=SHEN11_LOG_LAMBDA_MEAN,
    log_sigma=SHEN11_LOG_LAMBDA_SIGMA,
):
    """Logarithmically spaced Eddington ratio grid and its (unnormalised) pdf.

    Sampling on a log grid resolves the low-lambda end of the power law, which
    a linear grid under-samples for the steep default index.

    Two shapes are available. ``"power_law"`` is ``lambda**gamma_e``, the
    distribution of Korytov et al. (2019). Note that for the default
    ``gamma_e = -0.65`` this puts most of its probability *mass* near
    ``lambda = 1``: the mass per unit log(lambda) goes as
    ``lambda**(gamma_e + 1)``, which rises. ``"lognormal"`` is a Gaussian in
    log10(lambda), which is the shape observed for broad-line quasars and is
    centred well below Eddington.

    :param gamma_e: power-law index, for ``distribution="power_law"``
    :param lambda_min: minimum Eddington ratio
    :param lambda_max: maximum Eddington ratio
    :param n_grid: number of grid points
    :param distribution: "power_law" or "lognormal"
    :type distribution: str
    :param log_mean: mean of log10(lambda), for ``distribution="lognormal"``
    :param log_sigma: standard deviation of log10(lambda), for
        ``distribution="lognormal"``
    :return: grid of Eddington ratios, pdf evaluated on the grid
    :rtype: tuple of numpy.ndarray
    """
    grid = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_grid)
    if distribution == "power_law":
        return grid, grid**gamma_e
    if distribution == "lognormal":
        log_grid = np.log10(grid)
        density = np.exp(-0.5 * ((log_grid - log_mean) / log_sigma) ** 2)
        # convert the density in log10(lambda) to a density in lambda
        return grid, density / (grid * np.log(10))
    raise ValueError(
        "distribution must be 'power_law' or 'lognormal', got %s." % distribution
    )


def black_hole_mass_from_vel_disp(sigma_e, alpha=4.38, beta=0.310, scatter=0.0):
    """Black hole mass from the bulge velocity dispersion via the M-sigma
    relation of Kormendy & Ho (2013), https://arxiv.org/abs/1304.7762:

    .. math::
        \\frac{M_{BH}}{10^9 M_\\odot} = \\beta
        \\left(\\frac{\\sigma_e}{200\\,\\mathrm{km/s}}\\right)^{\\alpha}

    The relation is calibrated on ellipticals and classical bulges, and
    ``sigma_e`` is the *bulge* velocity dispersion. Applying it to a
    disc-dominated galaxy with a pseudobulge overestimates the black hole
    mass; restrict the host candidates to early-type galaxies if that matters.

    :param sigma_e: bulge velocity dispersion [km/s]
    :type sigma_e: float or numpy.ndarray
    :param alpha: power-law index
    :param beta: normalisation at 200 km/s, in units of 1e9 solar masses
    :param scatter: intrinsic scatter of the relation [dex]. The measured
        value is 0.29 dex; the default of zero returns the mean relation.
    :return: black hole mass [solar masses]
    :rtype: float or numpy.ndarray
    """
    log_mass = 9 + np.log10(beta) + alpha * np.log10(np.asarray(sigma_e) / 200)
    if scatter:
        log_mass = log_mass + np.random.normal(0, scatter, np.shape(log_mass))
    return 10**log_mass


def bolometric_luminosity_from_l3000(
    l3000, form="linear", anisotropy_correction=1.0, scatter=0.0
):
    """Bolometric luminosity from the monochromatic 3000 Angstrom luminosity.

    Uses the Runnoe et al. (2012) corrections,
    https://arxiv.org/abs/1201.5155 (with the erratum coefficients).

    :param l3000: monochromatic luminosity lambda L_lambda(3000 A) [erg/s]
    :type l3000: float or numpy.ndarray
    :param form: "linear" for L_iso = zeta * lambda L_lambda, or "log" for the
        non-linear log-log fit
    :type form: str
    :param anisotropy_correction: factor applied to the isotropic luminosity
        to account for the viewing angle of the disc. Runnoe et al. recommend
        0.75; the default of 1 matches the convention of the SDSS quasar
        property catalogs.
    :param scatter: object-to-object scatter of the correction [dex]
    :return: bolometric luminosity [erg/s]
    :rtype: float or numpy.ndarray
    """
    coefficients = RUNNOE12_BOLOMETRIC_CORRECTION[3000]
    log_l3000 = np.log10(l3000)
    if form == "linear":
        log_l_iso = log_l3000 + np.log10(coefficients["zeta"])
    elif form == "log":
        log_l_iso = coefficients["A"] + coefficients["B"] * log_l3000
    else:
        raise ValueError("form must be either 'linear' or 'log', got %s." % form)
    log_l_bol = log_l_iso + np.log10(anisotropy_correction)
    if scatter:
        log_l_bol = log_l_bol + np.random.normal(0, scatter, np.shape(log_l_bol))
    return 10**log_l_bol


@lru_cache(maxsize=8)
def _absolute_i_magnitude_reference(band="lsst2023-i"):
    """Absolute i-band magnitude of a qsogen quasar with a known 3000 A
    luminosity.

    A quasar SED is generated at z = 2 and normalised to
    ``10**_LOG_L3000_REFERENCE`` erg/s at 3000 Angstrom, and its absolute
    magnitude in the M_i(z=2) system of Richards et al. (2006) is measured by
    synthetic photometry. The luminosity distance used to flux-normalise the
    SED cancels against the distance modulus, so the result is a property of
    the SED alone and is independent of cosmology.

    :param band: speclite filter defining the i band
    :type band: str
    :return: absolute i-band magnitude at the reference luminosity
    :rtype: float
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    wavlen = np.logspace(2.0, 4.48, num=20001, endpoint=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sed = Quasar_sed(
            z=2.0,
            LogL3000=_LOG_L3000_REFERENCE,
            params=params_agile,
            wavlen=wavlen,
            cosmo=cosmo,
        )
    filters = speclite.filters.load_filters(band)
    apparent_mag = filters.get_ab_magnitudes(sed.flux, sed.wavred)[0][0]
    return float(apparent_mag - cosmo.distmod(2.0).value)


def l3000_from_absolute_i_magnitude(m_i, band="lsst2023-i"):
    """Monochromatic 3000 Angstrom luminosity from the absolute i-band
    magnitude.

    ``m_i`` is the K-corrected absolute i-band magnitude normalised to z = 2,
    M_i(z=2), which is the quantity the Richards et al. (2006) / Oguri &
    Marshall (2010) luminosity function is written in and the quantity qsogen
    expects. The conversion is a pure rescaling of the qsogen SED, so it is
    exactly consistent with the broad-band photometry generated from the same
    SED.

    :param m_i: absolute i-band magnitude M_i(z=2)
    :type m_i: float or numpy.ndarray
    :param band: speclite filter defining the i band
    :type band: str
    :return: lambda L_lambda(3000 A) [erg/s]
    :rtype: float or numpy.ndarray
    """
    reference = _absolute_i_magnitude_reference(band)
    return 10 ** (_LOG_L3000_REFERENCE + 0.4 * (reference - np.asarray(m_i)))


def absolute_i_magnitude_from_l3000(l3000, band="lsst2023-i"):
    """Inverse of :func:`l3000_from_absolute_i_magnitude`.

    :param l3000: lambda L_lambda(3000 A) [erg/s]
    :type l3000: float or numpy.ndarray
    :param band: speclite filter defining the i band
    :type band: str
    :return: absolute i-band magnitude M_i(z=2)
    :rtype: float or numpy.ndarray
    """
    reference = _absolute_i_magnitude_reference(band)
    return reference - 2.5 * (np.log10(l3000) - _LOG_L3000_REFERENCE)


# Effective wavelengths of the LSST bands [nm], from Fig. 1 of Huber et al.
# 2020, https://arxiv.org/abs/2008.10393.
_LSST_EFFECTIVE_WAVELENGTH_NM = {
    "u": 367.1,
    "g": 482.7,
    "r": 622.3,
    "i": 754.6,
    "z": 869.1,
    "y": 971.2,
}

# Bolometric corrections of Runnoe et al. (2012) mapped onto the LSST bands by
# rest-frame wavelength: u is closest to 3000 A, the rest to 5100 A.
_LSST_BOLOMETRIC_CORRECTION = {
    "u": RUNNOE12_BOLOMETRIC_CORRECTION[3000]["zeta"],
    "g": RUNNOE12_BOLOMETRIC_CORRECTION[5100]["zeta"],
    "r": RUNNOE12_BOLOMETRIC_CORRECTION[5100]["zeta"],
    "i": RUNNOE12_BOLOMETRIC_CORRECTION[5100]["zeta"],
    "z": RUNNOE12_BOLOMETRIC_CORRECTION[5100]["zeta"],
    "y": RUNNOE12_BOLOMETRIC_CORRECTION[5100]["zeta"],
}


def calculate_lsst_magnitude(lsst_band, black_hole_mass_Msun, eddington_ratio):
    """Rest-frame absolute AB magnitude of a quasar in an LSST band.

    The Eddington luminosity is set by the black hole mass, the bolometric
    luminosity by the Eddington ratio, and a grey bolometric correction
    converts it to the monochromatic luminosity of the band, which is then
    turned into an AB magnitude at the effective wavelength of that band.

    This is a coarse estimate that assumes the band luminosity is
    ``L_bol / BC`` at a single effective wavelength.
    :class:`QuasarHostMatch` uses the qsogen SED instead, via
    :func:`l3000_from_absolute_i_magnitude`, and should be preferred.

    :param lsst_band: one of 'u', 'g', 'r', 'i', 'z', 'y'
    :type lsst_band: str
    :param black_hole_mass_Msun: black hole mass [solar masses]
    :type black_hole_mass_Msun: float or numpy.ndarray
    :param eddington_ratio: Eddington ratio L_bol / L_Edd
    :type eddington_ratio: float or numpy.ndarray
    :return: absolute AB magnitude in the requested band
    :rtype: float or numpy.ndarray
    :raises ValueError: if the band is not an LSST band
    """
    if lsst_band not in _LSST_BOLOMETRIC_CORRECTION:
        raise ValueError(
            "Invalid LSST band '%s'. Must be one of %s."
            % (lsst_band, list(_LSST_BOLOMETRIC_CORRECTION.keys()))
        )

    l_bol = L_EDDINGTON_PER_MSUN * np.asarray(black_hole_mass_Msun) * eddington_ratio
    nu_l_nu = l_bol / _LSST_BOLOMETRIC_CORRECTION[lsst_band]

    # nu L_nu -> L_nu -> flux density at 10 pc -> AB magnitude
    frequency = 2.99792458e17 / _LSST_EFFECTIVE_WAVELENGTH_NM[lsst_band]  # Hz
    ten_parsec_cm = 3.0856775814913673e19
    flux_density = nu_l_nu / frequency / (4 * np.pi * ten_parsec_cm**2)
    return -2.5 * np.log10(flux_density) - 48.60


class QuasarHostMatch(object):
    """Assign host galaxies, black hole masses and Eddington ratios to a
    quasar catalog.

    For every quasar the bolometric luminosity implied by its absolute
    magnitude is computed, and a (host galaxy, Eddington ratio) pair is drawn
    from

    .. math::
        p(k, \\lambda) \\propto p(\\lambda)\\,
        \\mathcal{N}\\!\\left(\\log M_{BH}^{req}(\\lambda) \\,\\middle|\\,
        \\log M_{BH}(\\sigma_k),\\, s_{M\\sigma}\\right)

    where the required black hole mass is the one that reproduces the observed
    luminosity at that Eddington ratio, and the Gaussian is the intrinsic
    scatter of the M-sigma relation. Candidate hosts ``k`` are the galaxies in
    a thin redshift slice around the quasar, so a uniform prior over them
    correctly weights by the galaxy number density. The reported black hole
    mass is the required one, so the catalog satisfies
    ``L_bol = eddington_ratio * L_Edd(M_BH)`` exactly.
    """

    def __init__(
        self,
        quasar_catalog,
        galaxy_catalog,
        delta_z=0.001,
        min_candidates=50,
        max_delta_z=0.05,
        m_sigma_scatter=0.29,
        bolometric_correction_scatter=0.1,
        bolometric_correction_form="linear",
        anisotropy_correction=1.0,
        eddington_ratio_distribution="power_law",
        lambda_bounds=None,
        gamma_e=-0.65,
        eddington_log_mean=SHEN11_LOG_LAMBDA_MEAN,
        eddington_log_sigma=SHEN11_LOG_LAMBDA_SIGMA,
        n_eddington_grid=32,
        max_offset_sigma=4.0,
        unique_hosts=False,
        i_band="lsst2023-i",
        galaxy_types=None,
        progress=True,
    ):
        """

        :param quasar_catalog: quasar catalog with a redshift column "z" and an
            absolute i-band magnitude column "M_i" in the M_i(z=2) system
        :type quasar_catalog: astropy Table
        :param galaxy_catalog: host galaxy candidates, requiring "z" and
            "vel_disp" columns
        :type galaxy_catalog: astropy Table
        :param delta_z: half-width of the redshift slice from which host
            candidates are drawn
        :param min_candidates: the slice is widened until it holds at least
            this many galaxies, up to ``max_delta_z``
        :param max_delta_z: largest half-width the slice may be widened to
        :param m_sigma_scatter: intrinsic scatter of the M-sigma relation [dex]
        :param bolometric_correction_scatter: object-to-object scatter of the
            bolometric correction [dex]
        :param bolometric_correction_form: "linear" or "log", see
            :func:`bolometric_luminosity_from_l3000`
        :param anisotropy_correction: disc viewing-angle factor, see
            :func:`bolometric_luminosity_from_l3000`
        :param eddington_ratio_distribution: "power_law" for the shape of
            Korytov et al. (2019), or "lognormal" for a Gaussian in
            log10(lambda). The power law places most of its probability mass
            near the Eddington limit, well above what is observed for
            broad-line quasars; see :func:`eddington_ratio_grid`.
        :type eddington_ratio_distribution: str
        :param lambda_bounds: (min, max) Eddington ratio, defaulting to
            ``DEFAULT_LAMBDA_BOUNDS`` for the chosen distribution. The 0.01
            lower bound of the power law is roughly where a radiatively
            efficient thin disc gives way to an advection dominated flow, and
            matches the lower edge of ``agn_bounds_dict`` used by the
            variability model. Note that the 0.1 lower bound of Korytov et al.
            (2019) is too high for a luminosity function sampled far below the
            knee: a quasar with M_i = -19 needs a 1e6 solar mass black hole
            even at lambda = 0.1, and no galaxy in a typical catalog is that
            small, so those quasars would all be rejected.
        :param gamma_e: power-law index, for the "power_law" distribution
        :param eddington_log_mean: mean of log10(lambda), for the "lognormal"
            distribution
        :param eddington_log_sigma: standard deviation of log10(lambda), for
            the "lognormal" distribution
        :param n_eddington_grid: number of Eddington ratio grid points used in
            the weighted draw
        :param max_offset_sigma: a quasar is rejected if no candidate host can
            produce it within this many times ``m_sigma_scatter``
        :param unique_hosts: if True, a galaxy hosts at most one quasar
        :param i_band: speclite filter defining the i band
        :param galaxy_types: if given, restrict host candidates to these values
            of the "galaxy_type" column, e.g. ``["red"]`` to keep only
            bulge-dominated hosts, for which the M-sigma relation is calibrated
        :type galaxy_types: list of str or None
        :param progress: whether to show a progress bar
        """
        self.quasar_catalog = quasar_catalog
        self.galaxy_catalog = galaxy_catalog.copy()
        self._delta_z = delta_z
        self._min_candidates = min_candidates
        self._max_delta_z = max_delta_z
        self._m_sigma_scatter = m_sigma_scatter
        self._bc_scatter = bolometric_correction_scatter
        self._bc_form = bolometric_correction_form
        self._anisotropy_correction = anisotropy_correction
        self._max_offset_sigma = max_offset_sigma
        self._unique_hosts = unique_hosts
        self._i_band = i_band
        self._galaxy_types = galaxy_types
        self._progress = progress

        if lambda_bounds is None:
            lambda_bounds = DEFAULT_LAMBDA_BOUNDS[eddington_ratio_distribution]
        self._lambda_grid, pdf = eddington_ratio_grid(
            gamma_e=gamma_e,
            lambda_min=lambda_bounds[0],
            lambda_max=lambda_bounds[1],
            n_grid=n_eddington_grid,
            distribution=eddington_ratio_distribution,
            log_mean=eddington_log_mean,
            log_sigma=eddington_log_sigma,
        )
        # trapezoidal weights so the discrete grid integrates the pdf
        spacing = np.gradient(self._lambda_grid)
        self._lambda_weight = pdf * spacing
        self._lambda_weight /= self._lambda_weight.sum()
        self._log_lambda_grid = np.log10(self._lambda_grid)
        # half-width of a grid cell in log space, used to jitter the drawn
        # Eddington ratio within its cell so the output is not quantised
        self._log_lambda_half_width = 0.5 * np.gradient(self._log_lambda_grid)

        # number of quasars for which no host could produce the luminosity
        self.n_rejected = 0

    def match(self):
        """Match every quasar with a host galaxy.

        :return: catalog of the quasars that could be matched, joined with
            their host galaxies and with "black_hole_mass_exponent",
            "eddington_ratio" and "log_bolometric_luminosity" columns added
        :rtype: astropy Table
        """
        if "vel_disp" not in self.galaxy_catalog.colnames:
            raise ValueError(
                "Galaxy catalog must have 'vel_disp' column to perform quasar-host match."
            )
        for column in ["z", "M_i"]:
            if column not in self.quasar_catalog.colnames:
                raise ValueError(
                    "Quasar catalog must have a '%s' column to perform "
                    "quasar-host match." % column
                )
        if self._galaxy_types is not None:
            if "galaxy_type" not in self.galaxy_catalog.colnames:
                raise ValueError(
                    "Galaxy catalog must have a 'galaxy_type' column to select on it."
                )
            # FITS stores strings as bytes, so a catalog round-tripped through a
            # file has a bytes dtype here and would match nothing
            types = np.asarray(self.galaxy_catalog["galaxy_type"])
            if types.dtype.kind == "S":
                types = np.char.decode(types)
            keep = np.isin(types, self._galaxy_types)
            if not keep.any():
                raise ValueError(
                    "No galaxy in the catalog has a 'galaxy_type' among %s. "
                    "The catalog contains %s."
                    % (list(self._galaxy_types), sorted(set(types.tolist()))[:10])
                )
            self.galaxy_catalog = self.galaxy_catalog[keep]

        # galaxies with an unmeasured velocity dispersion carry no information
        # about the black hole mass and cannot be hosts
        self.galaxy_catalog = self.galaxy_catalog[self.galaxy_catalog["vel_disp"] > 0]

        self.galaxy_catalog.sort("z")
        # dtype=float forces native byte order: a catalog read from FITS is
        # big-endian, and numpy has no fast searchsorted path for that, which
        # costs ~25 ms per lookup instead of ~1 us
        galaxy_z = np.asarray(self.galaxy_catalog["z"], dtype=float)
        galaxy_vel_disp = np.asarray(self.galaxy_catalog["vel_disp"], dtype=float)
        log_mass_host = np.log10(black_hole_mass_from_vel_disp(galaxy_vel_disp))
        available = np.ones(len(galaxy_z), dtype=bool)

        # bolometric luminosity implied by the quasar magnitude, with the
        # scatter of the bolometric correction applied once per quasar
        l3000 = l3000_from_absolute_i_magnitude(
            np.asarray(self.quasar_catalog["M_i"], dtype=float), band=self._i_band
        )
        log_l_bol = np.log10(
            bolometric_luminosity_from_l3000(
                l3000,
                form=self._bc_form,
                anisotropy_correction=self._anisotropy_correction,
                scatter=self._bc_scatter,
            )
        )
        # black hole mass needed to radiate that luminosity at each grid lambda
        log_mass_required = (
            log_l_bol[:, None]
            - np.log10(L_EDDINGTON_PER_MSUN)
            - self._log_lambda_grid[None, :]
        )

        matched_galaxy_indices = []
        matched_quasar_indices = []
        matched_log_mass = []
        matched_eddington_ratio = []

        quasar_z = np.asarray(self.quasar_catalog["z"], dtype=float)
        rows = tqdm(
            range(len(self.quasar_catalog)),
            desc="Matching quasars with host galaxies",
            disable=not self._progress,
        )
        for i in rows:
            start, end = self._candidate_range(galaxy_z, quasar_z[i], available)
            if start is None:
                self.n_rejected += 1
                continue

            if self._unique_hosts:
                candidates = np.flatnonzero(available[start:end]) + start
                mass = log_mass_host[candidates]
            else:
                # every galaxy in the slice is available, so index it as a view
                # rather than materialising the indices for each quasar
                candidates = None
                mass = log_mass_host[start:end]

            required = log_mass_required[i]
            offset = (required[None, :] - mass[:, None]) / self._m_sigma_scatter
            if np.min(np.abs(offset)) > self._max_offset_sigma:
                self.n_rejected += 1
                continue

            weight = self._lambda_weight[None, :] * np.exp(-0.5 * offset**2)
            cumulative = np.cumsum(weight)
            if not cumulative[-1] > 0:
                self.n_rejected += 1
                continue

            flat = min(
                np.searchsorted(cumulative, np.random.uniform(0, 1) * cumulative[-1]),
                weight.size - 1,
            )
            host_local, lambda_index = np.unravel_index(flat, weight.shape)
            host_index = start + host_local if candidates is None else candidates[host_local]

            # spread the draw uniformly across the grid cell it landed in, and
            # take the black hole mass that reproduces the luminosity at that
            # exact Eddington ratio
            half_width = self._log_lambda_half_width[lambda_index]
            log_lambda = self._log_lambda_grid[lambda_index] + np.random.uniform(
                -half_width, half_width
            )
            matched_galaxy_indices.append(host_index)
            matched_quasar_indices.append(i)
            matched_log_mass.append(
                log_l_bol[i] - np.log10(L_EDDINGTON_PER_MSUN) - log_lambda
            )
            matched_eddington_ratio.append(10**log_lambda)
            if self._unique_hosts:
                available[host_index] = False

        matched_quasars = self.quasar_catalog[matched_quasar_indices]
        matched_galaxies = self.galaxy_catalog[matched_galaxy_indices]
        matched_galaxies["black_hole_mass_exponent"] = matched_log_mass
        matched_galaxies["eddington_ratio"] = matched_eddington_ratio
        matched_galaxies["log_bolometric_luminosity"] = log_l_bol[
            matched_quasar_indices
        ]
        matched_galaxies.remove_column("z")

        if self.n_rejected:
            warnings.warn(
                "%d of %d quasars could not be assigned a host galaxy whose "
                "velocity dispersion reproduces their luminosity within %g sigma "
                "of the M-sigma relation, and were dropped."
                % (self.n_rejected, len(self.quasar_catalog), self._max_offset_sigma),
                UserWarning,
            )

        return hstack(
            [matched_quasars, matched_galaxies],
            table_names=["quasar", "host_galaxy"],
            join_type="exact",
        )

    def _candidate_range(self, galaxy_z, redshift, available):
        """Bounds of the redshift slice the host candidates are drawn from.

        The slice is widened geometrically until it holds ``min_candidates``
        galaxies or reaches ``max_delta_z``.

        :param galaxy_z: sorted galaxy redshifts
        :param redshift: quasar redshift
        :param available: boolean mask of galaxies not yet used as hosts
        :return: (start, end) indices, or (None, None) if the slice stays empty
        """
        delta_z = self._delta_z
        while True:
            start = np.searchsorted(galaxy_z, redshift - delta_z, side="left")
            end = np.searchsorted(galaxy_z, redshift + delta_z, side="right")
            count = (
                int(np.count_nonzero(available[start:end]))
                if self._unique_hosts
                else end - start
            )
            if count >= self._min_candidates or delta_z >= self._max_delta_z:
                break
            delta_z = min(delta_z * 2, self._max_delta_z)
        return (start, end) if count else (None, None)
