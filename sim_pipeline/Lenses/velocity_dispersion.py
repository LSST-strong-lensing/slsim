import numpy as np
import scipy
from skypy.galaxies.redshift import redshifts_from_comoving_density
from skypy.utils.random import schechter

#  TODO: some functionality may directly be imported from skypy, once a version is released
#  from skypy.galaxies.velocity_dispersion import schechter_vdf


def vel_disp_sdss(sky_area, redshift, vd_min, vd_max, cosmology, noise=True):
    """
    velocity dispersion function in a cone matched by SDSS measurements

    Parameters
    ----------
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    redshift : `numpy.array`
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    vd_min, vd_max: float
        Lower and upper bounds of random variable x. Samples are drawn uniformly from bounds.
    cosmology : Cosmology
        Cosmology object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts, velocity dispersion : tuple of array_like
        Redshifts and velocity dispersion of the galaxy sample described by the Schechter
        velocity dispersion function.

    Notes
    -----
    The probability distribution function :math:`p(\sigma)` for velocity dispersion :math:`\sigma`
    can be described by a Schechter function (see eq. (4) in [1]_)

    .. math::

        \phi = \phi_* \left(\frac{\sigma}{\sigma_*}\right)^\alpha
            \exp\left[-\left( \frac{\sigma}{\sigma_*} \right)^\beta\right]
            \frac{\beta}{\Gamma(\alpha/\beta)} \frac{1}{\sigma} \mathrm{d}\sigma \;.

    where :math:`\Gamma` is the gamma function, :math:`\sigma_*` is the
    characteristic velocity dispersion, :math:`\phi_*` is
    number density and
    :math:`\alpha` and :math:`\beta` are free parameters.

    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060

    """
    # SDSS velocity dispersion function for galaxies brighter than Mr >= -16.8
    phi_star = 8.0 * 10 ** (-3) / cosmology.h**3
    vd_star = 161
    alpha = 2.32
    beta = 2.67
    return schechter_vel_disp(redshift, phi_star, alpha, beta, vd_star, vd_min, vd_max, sky_area, cosmology,
                              noise=noise)


def schechter_vel_disp(redshift, phi_star, alpha, beta, vd_star, vd_min, vd_max, sky_area, cosmology, noise=True):
    r'''Sample redshifts and stellar masses from a Schechter mass function.

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
    vd_min, vd_max: int
        Lower and upper bounds of random variable x. Samples are drawn uniformly from bounds.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to calculate comoving densities.
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
        Redshifts and velocity dispersion of the galaxy sample described by the Schechter
        velocity dispersion function.

    '''

    # sample halo redshifts
    z = schechter_vel_disp_redshift(redshift, phi_star, alpha, beta, vd_star, vd_min, vd_max,
                                    sky_area, cosmology, noise)
    # sample galaxy mass for redshifts
    vel_disp = schechter_vdf(alpha, beta, vd_star, vd_min=vd_min, vd_max=vd_max, size=len(z), resolution=100)
    return z, vel_disp


def schechter_vel_disp_redshift(redshift, phi_star, alpha, beta, vd_star, vd_min, vd_max, sky_area,
                                cosmology, noise=True):
    r'''Sample redshifts from Schechter function.

    Sample the redshifts of velocity dispersion following a Schechter function
    with potentially redshift-dependent parameters, limited by velocity dispersion
    `vd_max` and `vd_min`, for a sky area `sky_area`.

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
    vd_min, vd_max: int
        Lower and upper bounds of random variable x. Samples are drawn uniformly from bounds.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmology : Cosmology
        Cosmology object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    velocity_dispersion: array_like
        Velocity dispersion drawn from Schechter function.

    Notes
    -----
    The probability distribution function :math:`p(\sigma)` for velocity dispersion :math:`\sigma`
    can be described by a Schechter function (see eq. (4) in [1]_)

    .. math::

        \phi = \phi_* \left(\frac{\sigma}{\sigma_*}\right)^\alpha
            \exp\left[-\left( \frac{\sigma}{\sigma_*} \right)^\beta\right]
            \frac{\beta}{\Gamma(\alpha/\beta)} \frac{1}{\sigma} \mathrm{d}\sigma \;.

    where :math:`\Gamma` is the gamma function, :math:`\sigma_*` is the
    characteristic velocity dispersion, :math:`\phi_*` is
    number density and
    :math:`\alpha` and :math:`\beta` are free parameters.

    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060


    '''
    alpha_prime = alpha / beta - 1
    x_min, x_max = (vd_min / vd_star) ** beta, (vd_max / vd_star) ** beta

    lnxmin = np.log(x_min)
    lnxmax = np.log(x_max)

    # gamma function integrand
    def f(lnx, a):
        return np.exp(a*lnx - np.exp(lnx)) if lnx < lnxmax.max() else 0.

    # integrate gamma function for each redshift

    gamma_ab = scipy.special.gamma(alpha/beta)

    gam = np.empty_like(redshift)

    for i, _ in np.ndenumerate(gam):

        gam[i], _ = scipy.integrate.quad(f, lnxmin, lnxmax, args=(alpha_prime,))

    # comoving number density is normalisation times upper incomplete gamma
    density = phi_star*gam / gamma_ab

    # sample redshifts from the comoving density
    return redshifts_from_comoving_density(redshift=redshift, density=density,
                                           sky_area=sky_area, cosmology=cosmology, noise=noise)


def schechter_vdf(alpha, beta, vd_star, vd_min, vd_max, size=None, resolution=1000):
    r"""Sample velocity dispersion of elliptical galaxies in the local universe
    following a Schecter function.

    Parameters
    ----------
    alpha: float
        The alpha parameter in the modified Schechter equation.
    beta: float
        The beta parameter in the modified Schechter equation.
    vd_star: float
        The characteristic velocity dispersion.
    vd_min, vd_max: int
        Lower and upper bounds of random variable x. Samples are drawn uniformly from bounds.
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
    The probability distribution function :math:`p(\sigma)` for velocity dispersion :math:`\sigma`
    can be described by a Schechter function (see eq. (4) in [1]_)

    .. math::

        \phi = \phi_* \left(\frac{\sigma}{\sigma_*}\right)^\alpha
            \exp\left[-\left( \frac{\sigma}{\sigma_*} \right)^\beta\right]
            \frac{\beta}{\Gamma(\alpha/\beta)} \frac{1}{\sigma} \mathrm{d}\sigma \;.

    where :math:`\Gamma` is the gamma function, :math:`\sigma_*` is the
    characteristic velocity dispersion, :math:`\phi_*` is
    number density of all spiral galaxies and
    :math:`\alpha` and :math:`\beta` are free parameters.

    References
    ----------
    .. [1] Choi, Park and Vogeley, (2007), astro-ph/0611607, doi:10.1086/511060

    """

    if np.ndim(alpha) > 0:
        raise NotImplementedError('only scalar alpha is supported')

    alpha_prime = alpha/beta - 1
    x_min, x_max = (vd_min/vd_star)**beta, (vd_max/vd_star)**beta

    samples = schechter(alpha_prime, x_min, x_max, resolution=resolution, size=size)
    samples = samples**(1/beta) * vd_star

    return samples
