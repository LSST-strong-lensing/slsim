import scipy as sp
import numpy as np
import astropy.constants as cnst

from typing import Union, Callable
from numpy.typing import ArrayLike
from astropy.units import Quantity
from astropy.cosmology import Cosmology
from skypy.utils.random import schechter
from skypy.galaxies.redshift import redshifts_from_comoving_density
from lenstronomy.Util.constants import arcsec
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.GalKin.numeric_kinematics import NumericKinematics


def schechter_vel_disp_redshift(
    redshift: ArrayLike,
    phi_star: Union[ArrayLike, Callable],
    alpha: float,
    beta: float,
    vd_star: float,
    vd_min: float,
    vd_max: float,
    sky_area: Quantity,
    cosmology: Cosmology,
    noise: bool = True,
) -> ArrayLike:
    r"""Sample redshifts from Schechter function.

    Sample the redshifts of velocity dispersion following a Schechter function
    with potentially redshift-dependent parameters, limited by velocity dispersion
    `vd_max` and `vd_min`, for a sky area `sky_area`.

    Parameters
    ----------
    redshift : ArrayLike
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    phi_star : Union[ArrayLike, Callable]
        Normalisation of the Schechter function. Can be a single value, an
        array of values for each `redshift`, or a function of redshift.
    alpha: float
        The alpha parameter in the modified Schechter equation.
    beta: float
        The beta parameter in the modified Schechter equation.
    vd_star: float
        The characteristic velocity dispersion.
    vd_min, vd_max: float
        Lower and upper bounds of random variable x. Samples are drawn uniformly from
        bounds.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : `~astropy.cosmology.Cosmology`
        Cosmology object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    velocity_dispersion: ArrayLike
        Velocity dispersion drawn from Schechter function.

    Notes
    -----
    The probability distribution function :math:`p(\\sigma)` for velocity dispersion
    :math:`\sigma` can be described by a Schechter function (see eq. (4) in [2]_)

    .. math::

        \\phi = \\phi_* \\left(\\frac{\\sigma}{\\sigma_*}\\right)^\\alpha
            \\exp\\left[-\\left( \\frac{\\sigma}{\\sigma_*} \\right)^\\beta\\right]
            \\frac{\\beta}{\\Gamma(\\alpha/\\beta)} \\frac{1}{\\sigma} \\mathrm{d}
            \\sigma \\;.

    where :math:`\\Gamma` is the gamma function, :math:`\\sigma_*` is the
    characteristic velocity dispersion, :math:`\\phi_*` is
    number density and
    :math:`\\alpha` and :math:`\beta` are free parameters.

    References
    ----------
    .. [2] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060
    """
    alpha_prime = alpha / beta - 1
    x_min, x_max = (vd_min / vd_star) ** beta, (vd_max / vd_star) ** beta

    lnxmin = np.log(x_min)
    lnxmax = np.log(x_max)

    # gamma function integrand
    def f(lnx, a):
        return (
            np.exp(lnx) * np.exp(a * lnx - np.exp(lnx)) if lnx < lnxmax.max() else 0.0
        )

    # integrate gamma function for each redshift

    gamma_ab = sp.special.gamma(alpha / beta)

    gam = np.empty_like(redshift)

    for i, _ in np.ndenumerate(gam):
        gam[i], _ = sp.integrate.quad(f, lnxmin, lnxmax, args=(alpha_prime,))

    # comoving number density is normalisation times upper incomplete gamma
    density = phi_star * gam / gamma_ab

    # sample redshifts from the comoving density
    return redshifts_from_comoving_density(
        redshift=redshift,
        density=density,
        sky_area=sky_area,
        cosmology=cosmology,
        noise=noise,
    )


def schechter_velocity_dispersion_function(
    alpha, beta, vd_star, vd_min, vd_max, size=None, resolution=1000
):
    """Sample velocity dispersion of elliptical galaxies in the local universe following
    a Schecter function.

    Parameters
    ----------
    alpha: float
        The alpha parameter in the modified Schechter equation.
    beta: float
        The beta parameter in the modified Schechter equation.
    vd_star: float
        The characteristic velocity dispersion.
    vd_min, vd_max: float
        Lower and upper bounds of random variable x. Samples are drawn uniformly from
        bounds.
    resolution: int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int
        Number of samples returned. Default is 1.

    Returns
    -------
    velocity_dispersion: array_like
        Velocity dispersion drawn from Schechter function.

    Notes
    -----
    The probability distribution function :math:`p(\\sigma)` for velocity dispersion
    :math:`\\sigma` can be described by a Schechter function (see eq. (4) in [3]_)

    .. math::

        \\phi = \\phi_* \\left(\\frac{\\sigma}{\\sigma_*}\\right)^\\alpha
            \\exp\\left[-\\left( \\frac{\\sigma}{\\sigma_*} \\right)^\\beta\\right]
            \\frac{\\beta}{\\Gamma(\\alpha/\\beta)} \frac{1}{\\sigma} \\mathrm{d}
            \\sigma \\;.

    where :math:`\\Gamma` is the gamma function, :math:`\\sigma_*` is the
    characteristic velocity dispersion, :math:`\\phi_*` is
    number density of all spiral galaxies and
    :math:`\\alpha` and :math:`\\beta` are free parameters.

    References
    ----------
    .. [3] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060
    """

    if np.ndim(alpha) > 0:
        raise NotImplementedError("only scalar alpha is supported")

    alpha_prime = alpha / beta - 1
    x_min, x_max = (vd_min / vd_star) ** beta, (vd_max / vd_star) ** beta

    samples = schechter(alpha_prime, x_min, x_max, resolution=resolution, size=size)
    samples = samples ** (1 / beta) * vd_star

    return samples


def schechter_vel_disp(
    redshift,
    vd_min,
    vd_max,
    sky_area,
    cosmology,
    phi_star=None,
    alpha=2.32,
    beta=2.67,
    vd_star=161.0,
    noise=True,
):
    r"""Sample redshifts and stellar masses from a Schechter mass function.

    Sample the redshifts and stellar masses of galaxies following a Schechter
    mass function with potentially redshift-dependent parameters, limited
    by maximum and minimum masses `m_min`, `m_max`, for a sky area `sky_area`.

    Parameters
    ----------
    redshift : array_like
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    phi_star : array_like or function
        Normalisation of the Schechter function. Can be a single value, an
        array of values for each `redshift`, or a function of redshift.
    alpha: float
        The alpha parameter in the modified Schechter equation.
    beta: float
        The beta parameter in the modified Schechter equation.
    vd_star: float
        The characteristic velocity dispersion.
    vd_min, vd_max: float
        Lower and upper bounds of random variable x. Samples are drawn uniformly from
        bounds.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : `~astropy.cosmology`
        `astropy.cosmology` object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Notes
    -----
    Effectively calls `~skypy.galaxies.redshift.schechter_smf_redshift` and
    `~skypy.galaxies.stellar_mass.schechter_smf_mass` internally and returns
    the tuple of results.

    Returns
    -------
    redshifts, velocity dispersion : tuple of array_like
        Redshifts and velocity dispersion of the galaxy sample described by the
        Schechter velocity dispersion function.
    """

    # sample halo redshifts
    z = schechter_vel_disp_redshift(
        redshift,
        phi_star,
        alpha,
        beta,
        vd_star,
        vd_min,
        vd_max,
        sky_area,
        cosmology,
        noise,
    )
    # sample galaxy mass for redshifts
    vel_disp = schechter_velocity_dispersion_function(
        alpha, beta, vd_star, vd_min=vd_min, vd_max=vd_max, size=len(z), resolution=100
    )
    return z, vel_disp


def vel_disp_from_m_star(m_star: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the velocity dispersion of a galaxy from its stellar mass using an
    empirical power-law relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::
        V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
        M_\\odot} \\right)^{0.24}

    Values taken from table 2 of [1]

    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    Args:
        m_star (Union[float, np.ndarray]): Stellar mass of the galaxy in solar masses.

    Returns:
        Union[float, np.ndarray]: Velocity dispersion in km/s.
    """

    v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)

    return v_disp
