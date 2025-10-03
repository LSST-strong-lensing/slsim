import numpy as np
import scipy
from scipy import interpolate
import copy

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion
from slsim.Util.param_util import vel_disp_from_m_star

"""
This module provides functions to compute velocity dispersion using schechter function.
"""
#  TODO: some functionality may directly be imported from skypy, once a version is
#  released
#  from skypy.galaxies.velocity_dispersion import schechter_vdf


def vel_disp_composite_model(r, m_star, rs_star, m_halo, c_halo, cosmo, z_lens):
    """Computes the luminosity weighted velocity dispersion for a deflector
    with a stellar Hernquist profile and a NFW halo profile, assuming isotropic
    anisotropy.

    :param r: radius of the luminosity-weighted velocity dispersion
        [arcsec]
    :param m_star: stellar mass [M_sun]
    :param rs_star: stellar half light radius [physical Mpc]
    :param m_halo: Halo mass [physical M_sun]
    :param c_halo: halo concentration
    :param cosmo: cosmology
    :type cosmo: ~astropy.cosmology class
    :param z_lens: redshift of the deflector
    :return: velocity dispersion [km/s]
    """

    kwargs_model = {
        "mass_profile_list": ["HERNQUIST", "NFW"],
        "light_profile_list": ["HERNQUIST"],
        "anisotropy_model": "const",
    }

    # turn physical masses to lenstronomy units
    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=10, cosmo=cosmo)
    # Hernquist profile
    sigma0, rs_angle_hernquist = lens_cosmo.hernquist_phys2angular(
        mass=m_star, rs=rs_star
    )
    # NFW profile
    rs_angle_nfw, alpha_Rs = lens_cosmo.nfw_physical2angle(M=m_halo, c=c_halo)
    kwargs_mass = [
        {"sigma0": sigma0, "Rs": rs_angle_hernquist, "center_x": 0, "center_y": 0},
        {"alpha_Rs": alpha_Rs, "Rs": rs_angle_nfw, "center_x": 0, "center_y": 0},
    ]
    kwargs_light = [{"amp": 1, "Rs": rs_angle_hernquist, "center_x": 0, "center_y": 0}]
    kwargs_anisotropy = {"beta": 0}

    from lenstronomy.GalKin.numeric_kinematics import NumericKinematics

    kwargs_numerics = {
        "interpol_grid_num": 1000,
        "log_integration": True,
        "max_integrate": 1000,
        "min_integrate": 0.0001,
        "max_light_draw": None,
        "lum_weight_int_method": True,
    }

    kwargs_cosmo = {"d_d": lens_cosmo.dd, "d_s": lens_cosmo.ds, "d_ds": lens_cosmo.dds}

    num_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)
    vel_disp = num_kin.lum_weighted_vel_disp(
        r, kwargs_mass, kwargs_light, kwargs_anisotropy
    )
    return vel_disp


def vel_disp_power_law(
    theta_E, gamma, r_half, kwargs_light, light_model_list, lens_cosmo
):
    """Velocity dispersion for a power-law mass density profile.

    :param theta_E: Einstein radius [arc seconds]
    :param gamma: power-law slope of deflector
    :param r_half: half light radius of deflector
    :param kwargs_light: list of dict for light model parameters
    :param light_model_list: list of light models
    :param lens_cosmo: ~LensCosmo instance
    :return: half light radius averaged velocity dispersion
    """
    # turn physical masses to lenstronomy units

    kwargs_mass = [
        {
            "theta_E": theta_E,
            "gamma": gamma,
            "e1": 0,
            "e2": 0,
            "center_x": 0,
            "center_y": 0,
        },
    ]
    mag2amp = MagAmpConversion(
        kwargs_model={"lens_light_model_list": light_model_list},
        magnitude_zero_point=30,
    )
    kwargs_light_amp, _, _ = mag2amp.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_light
    )

    kwargs_anisotropy = {"beta": 0}
    light_model = LightModel(light_model_list=light_model_list)
    lensLightProfile = LightProfileAnalysis(light_model=light_model)

    (
        amps,
        sigmas,
        center_x,
        center_y,
    ) = lensLightProfile.multi_gaussian_decomposition(
        kwargs_light_amp,
        r_h=r_half,
        n_comp=20,
    )
    light_profile_list = ["MULTI_GAUSSIAN"]
    kwargs_model = {
        "mass_profile_list": ["EPL"],
        "light_profile_list": light_profile_list,
        "anisotropy_model": "const",
    }

    kwargs_light_mge = [{"amp": amps, "sigma": sigmas}]

    from lenstronomy.GalKin.numeric_kinematics import NumericKinematics

    kwargs_numerics = {
        "interpol_grid_num": 1000,
        "log_integration": True,
        "max_integrate": 1000,
        "min_integrate": 0.0001,
        "max_light_draw": None,
        "lum_weight_int_method": True,
    }

    kwargs_cosmo = {"d_d": lens_cosmo.dd, "d_s": lens_cosmo.ds, "d_ds": lens_cosmo.dds}

    num_kin = NumericKinematics(kwargs_model, kwargs_cosmo, **kwargs_numerics)
    vel_disp = num_kin.lum_weighted_vel_disp(
        r_half, kwargs_mass, kwargs_light_mge, kwargs_anisotropy
    )
    return vel_disp


def theta_E_from_vel_disp_epl(
    vel_disp,
    gamma,
    r_half,
    kwargs_light,
    light_model_list,
    lens_cosmo,
    kappa_ext=0,
    sis_convention=True,
):
    """Calculates Einstein radius given measured aperture averaged velocity
    dispersion and given power-law slope.

    :param vel_disp: velocity dispersion measured within an aperture
        radius [km/s]
    :param gamma: power-law slope
    :param r_half: half light radius (aperture radius) [arc seconds]
    :param kwargs_light: list of dict for light model parameters
    :param light_model_list: list of light models
    :param lens_cosmo: ~LensCosmo instance
    :param kappa_ext: external convergence
    :param sis_convention: it True, uses velocity dispersion not as
        measured one but as the SIS equivalent velocity dispersion
    :return: Einstein radius matching the velocity dispersion and
        external convergence
    """
    if gamma == 2 or sis_convention is True:
        theta_E = lens_cosmo.sis_sigma_v2theta_E(vel_disp)
    else:
        theta_E_0 = 1
        vel_disp_0 = vel_disp_power_law(
            theta_E_0, gamma, r_half, kwargs_light, light_model_list, lens_cosmo
        )
        # transform theta_E from vel_disp_0 prediction
        theta_E = theta_E_0 * (vel_disp / vel_disp_0) ** (2 / (gamma - 1))
    theta_E /= (1 - kappa_ext) ** (1.0 / (gamma - 1))
    return theta_E


def vel_disp_nfw_3d(r, m_halo, c_halo, cosmo, z_lens):
    """Computes the unweighted velocity dispersion at 3D radius r for a
    deflector with a NFW halo profile, assuming isotropic anisotropy (beta =
    0).

    Based on equation (14) of Lokas and Mamon 2001 (
    https://arxiv.org/abs/astro-ph/0002395)

    :param r: radius of the unweighted velocity dispersion [arcsec]
    :param m_halo: Halo mass [physical M_sun]
    :param c_halo: halo concentration
    :param cosmo: cosmology
    :type cosmo: ~astropy.cosmology class
    :param z_lens: redshift of the deflector
    :return: velocity dispersion [km/s]
    """

    from scipy.special import spence
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    from astropy.constants import G
    from lenstronomy.Util.constants import arcsec

    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=10, cosmo=cosmo)
    r_halo = lens_cosmo.nfw_M_theta_r200(m_halo)
    s = r / r_halo
    vel2 = (
        G.to("km2 Mpc / M_sun s2").value * m_halo / (r_halo * arcsec * lens_cosmo.dd)
    )  # km^2 / s^2
    cs = c_halo * s
    g_c = 1 / (np.log(1 + c_halo) - c_halo / (1 + c_halo))
    vel_disp2 = vel2 * (
        1
        / 2
        * c_halo**2
        * g_c
        * s
        * (1 + cs) ** 2
        * (
            np.pi**2
            - np.log(cs)
            - 1 / cs
            - 1 / (1 + cs) ** 2
            - 6 / (1 + cs)
            + (1 + 1 / cs**2 - 4 / cs - 2 / (1 + cs)) * np.log(1 + cs)
            + 3 * np.log(1 + cs) ** 2
            + 6 * spence(1 + cs)
        )
    )
    return np.sqrt(vel_disp2)


def vel_disp_nfw_aperture(r, m_halo, c_halo, cosmo, z_lens):
    """Computes the average line-of-sight velocity dispersion in an aperture r
    for a deflector with a NFW halo profile, assuming isotropic anisotropy
    (beta = 0).

    Based on equation (48) of Lokas & Mamon 2001 (
    https://arxiv.org/abs/astro-ph/0002395)

    :param r: radius of the aperture for the velocity dispersion
        [arcsec]
    :param m_halo: Halo mass [physical M_sun]
    :param c_halo: halo concentration
    :param cosmo: cosmology
    :type cosmo: ~astropy.cosmology class
    :param z_lens: redshift of the deflector
    :return: velocity dispersion [km/s]
    """
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    from lenstronomy.LensModel.Profiles.nfw import NFW

    def _log_integrate(func, xmin, xmax, n_grid=200):
        min_log = np.log(xmin)
        max_log = np.log(xmax)
        dlogx = (max_log - min_log) / (n_grid - 1)
        x = np.logspace(
            min_log + dlogx / 2.0,
            max_log + dlogx / 2.0,
            n_grid,
            base=np.e,
        )
        dlog_x = np.log(x[2]) - np.log(x[1])
        y = func(x)
        return np.sum(y * dlog_x * x)

    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=10, cosmo=cosmo)

    g_c = 1 / (np.log(1 + c_halo) - c_halo / (1 + c_halo))
    rs, alpha_rs = lens_cosmo.nfw_physical2angle(m_halo, c_halo)
    r_halo = rs * c_halo
    rmin = 1e-3 * rs
    rmax = 1e3 * rs
    nfw = NFW()
    m_2d_r = nfw.mass_2d_lens(r, rs, alpha_rs) * lens_cosmo.sigma_crit_angle
    int1 = _log_integrate(
        lambda r_: vel_disp_nfw_3d(r_, m_halo, c_halo, cosmo, z_lens) ** 2
        * r_
        / r_halo
        / (1 + c_halo * r_ / r_halo) ** 2,
        rmin,
        rmax,
    )
    int2 = _log_integrate(
        lambda r_: (
            vel_disp_nfw_3d(r_, m_halo, c_halo, cosmo, z_lens) ** 2
            / (1 + c_halo * r_ / r_halo) ** 2
            * np.sqrt((r_ / r_halo) ** 2 - (r / r_halo) ** 2)
        ),
        r,
        rmax,
    )
    vel_disp2 = c_halo**2 * g_c * m_halo / m_2d_r * (int1 - int2) / r_halo
    return np.sqrt(vel_disp2)


def vel_disp_nfw(m_halo, c_halo, cosmo, z_lens):
    """Computes vel_disp_nfw_aperture using the characteristic radius rs of the
    NFW as aperture (which is independent of the source redshift).

    :param m_halo: Halo mass [physical M_sun]
    :param c_halo: halo concentration
    :param cosmo: cosmology
    :type cosmo: ~astropy.cosmology class
    :param z_lens: redshift of the deflector
    :return: velocity dispersion [km/s]
    """
    from lenstronomy.Cosmo.lens_cosmo import LensCosmo

    lens_cosmo = LensCosmo(z_lens=z_lens, z_source=10, cosmo=cosmo)

    rs_arcsec, _ = lens_cosmo.nfw_physical2angle(m_halo, c_halo)
    vel_disp = vel_disp_nfw_aperture(
        r=rs_arcsec,
        m_halo=m_halo,
        c_halo=c_halo,
        cosmo=cosmo,
        z_lens=z_lens,
    )
    return vel_disp


def vel_disp_sdss(sky_area, redshift, vd_min, vd_max, cosmology, noise=True):
    """Velocity dispersion function in a cone matched by SDSS measurements.

    Parameters
    ----------
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    redshift : `numpy.array`
        Input redshift grid on which the Schechter function parameters are
        evaluated. Galaxies are sampled over this redshift range.
    vd_min, vd_max: int
        Lower and upper bounds of random variable x (velocity dispersion). Samples are
        drawn uniformly from bounds.
    cosmology : `astropy.cosmology`
        `astropy.cosmology` object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts, velocity dispersion : tuple of array_like
        Redshifts and velocity dispersion of the galaxy sample described by the
        Schechter velocity dispersion function.

    Notes
    -----
    The probability distribution function :math:`p(\\sigma)` for velocity dispersion
    :math:`\\sigma` can be described by a Schechter function (see eq. (4) in [1]_)

    .. math::
        \\phi = \\phi_* \\left(\\frac{\\sigma}{\\sigma_*}\\right)^\\alpha
        \\exp\\left[-\\left( \\frac{\\sigma}{\\sigma_*} \\right)^\\beta\\right]
        \\frac{\\beta}{\\Gamma(\\alpha/\\beta)} \frac{1}{\\sigma}
        \\mathrm{d}\\sigma \\;.

    where :math:`\\Gamma` is the gamma function, :math:`\\sigma_*` is the
    characteristic velocity dispersion, :math:`\\phi_*` is
    number density and
    :math:`\\alpha` and :math:`\\beta` are free parameters.

    References
    ----------
    .. [1] Bernardi et al. 2010,
     https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.2087B/abstract
    """
    # SDSS velocity dispersion function for galaxies brighter than Mr >= -16.8
    # These numbers are from the Bernardi et al. 2010.
    phi_star = 2.099e-2 * (cosmology.h / 0.7) ** 3
    vd_star = 113.78
    alpha = 0.94
    beta = 1.85
    return schechter_vel_disp(
        redshift,
        phi_star,
        alpha,
        beta,
        vd_star,
        vd_min,
        vd_max,
        sky_area,
        cosmology,
        noise=noise,
    )


def schechter_vel_disp(
    redshift,
    phi_star,
    alpha,
    beta,
    vd_star,
    vd_min,
    vd_max,
    sky_area,
    cosmo,
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
    cosmo : `~astropy.cosmology`
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
        cosmo,
        noise=noise,
    )
    # sample galaxy mass for redshifts
    vel_disp = schechter_velocity_dispersion_function(
        alpha,
        beta,
        phi_star,
        vd_star,
        vd_min,
        vd_max,
        size=len(z),
        resolution=1000,
        scale=1,
    )
    # above vel_disp is measured with in the effective radius/8. So, correction factor
    # of 8**(-0.066) must be applied to convert to vel_disp at effective radius.
    # This is from Cappellari et al. (2006): 10.1111/j.1365-2966.2005.09981.x
    sigma_e = vel_disp * (8 ** (-0.066))
    return z, sigma_e


def schechter_vel_disp_redshift(
    redshift,
    phi_star,
    alpha,
    beta,
    vd_star,
    vd_min,
    vd_max,
    sky_area,
    cosmo,
    noise=True,
):
    r"""Sample redshifts from Schechter function.

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
    vd_min, vd_max: float
        Lower and upper bounds of random variable x. Samples are drawn uniformly from
        bounds.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmo : `~astropy.cosmology`
        `astropy.cosmology` object to calculate comoving densities.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts: array_like
        redshifts drawn from Schechter function.

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
    """alpha_prime = alpha / beta - 1
    x_min, x_max = (vd_min / vd_star) ** beta, (vd_max / vd_star) ** beta

    lnxmin = np.log(x_min)
    lnxmax = np.log(x_max)"""

    # gamma function integrand
    def f(v):
        return (
            phi_star
            * ((v / vd_star) ** alpha)
            * np.exp(-((v / vd_star) ** beta))
            * beta
            / v
        )

    # integrate gamma function for each redshift

    gamma_ab = scipy.special.gamma(alpha / beta)

    gam = np.empty_like(redshift)

    for i, _ in np.ndenumerate(gam):
        gam[i], _ = scipy.integrate.quad(f, vd_min, vd_max)

    # comoving number density is normalisation times upper incomplete gamma
    density = gam / gamma_ab

    # sample redshifts from the comoving density
    return redshifts_from_comoving_density(
        redshift, density, sky_area, cosmo, noise=noise
    )


def redshifts_from_comoving_density(redshift, density, sky_area, cosmo, noise=True):
    r"""Sample redshifts from a comoving density function. We took this function
    is from SkyPy package but we have modified it to make suitable for the
    constant comoving number density.

    Sample galaxy redshifts such that the resulting distribution matches a past
    lightcone with comoving galaxy number density `density` at redshifts
    `redshift`. The comoving volume sampled corresponds to a sky area `sky_area`
    and transverse comoving distance given by the cosmology `cosmology`.

    If the `noise` parameter is set to true, the number of galaxies has Poisson
    noise. If `noise` is false, the expected number of galaxies is used.

    Parameters
    ----------
    redshift : array_like
        Redshifts at which comoving number densities are provided.
    density : array_like
        Comoving galaxy number density at each redshift in Mpc-3.
    sky_area : `~astropy.units.Quantity`
        Sky area over which galaxies are sampled. Must be in units of solid angle.
    cosmo : Cosmology
        Cosmology object for conversion to comoving volume.
    noise : bool, optional
        Poisson-sample the number of galaxies. Default is `True`.

    Returns
    -------
    redshifts : array_like
        Sampled redshifts such that the comoving number density of galaxies
        corresponds to the input distribution.

    Warnings
    --------
    The inverse cumulative distribution function is approximated from the
    number density and comoving volume calculated at the given `redshift`
    values. The user must choose suitable `redshift` values to satisfy their
    desired numerical accuracy.
    """

    # redshift number density
    dN_dz = (cosmo.differential_comoving_volume(redshift) * sky_area).to_value("Mpc3")
    dN_dz *= density
    # number
    N = np.trapz(dN_dz, redshift)
    # Poisson sample galaxy number if requested
    if noise:
        total_number = np.random.poisson(N)
    else:
        total_number = int(N)  # np.array([int(digit) for digit in number])
    cdf = dN_dz  # in place
    np.cumsum((dN_dz[1:] + dN_dz[:-1]) / 2 * np.diff(redshift), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]
    return np.interp(np.random.rand(total_number), cdf, redshift)


def schechter_velocity_dispersion_function(
    alpha, beta, phi_star, vd_star, vd_min, vd_max, size=None, resolution=1000, scale=1
):
    """Sample velocity dispersion of elliptical galaxies in the local universe
    following a Schecter function.

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
    size: int
        Number of samples returned. Default is 1.
    resolution: int
        Resolution of the inverse transform sampling spline. Default is 100.
    scale: array-like, optional
        Scale factor for the returned samples. Default is 1.

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

    if size is None:
        size = np.broadcast(vd_min, vd_max, scale).shape or None

    v = np.linspace(vd_min, vd_max, resolution)
    gamma_ab = scipy.special.gamma(alpha / beta)
    phi_star = phi_star
    pdf = (
        phi_star
        * ((v / vd_star) ** alpha)
        * np.exp(-((v / vd_star) ** beta))
        * beta
        / (v * gamma_ab)
    )
    cdf = pdf  # in place
    np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(v), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    t_lower = np.interp(vd_min, v, cdf)
    t_upper = np.interp(vd_max, v, cdf)
    u = np.random.uniform(t_lower, t_upper, size=size)
    v_sample = np.interp(u, cdf, v)
    return v_sample


def vel_disp_abundance_matching(galaxy_list, z_max, sky_area, cosmo):
    """Calculates the velocity dispersion from the steller mass. The routine
    uses abundance matching between stellar mass and velocity dispersion taking
    the sample drawn from z=0 to z_max (which can be still at low redshift
    where there is data on the velocity dispersion function)

    :param galaxy_list: list of galaxies with stellar masses given
    :type galaxy_list: ~astropy.Table object
    :param z_max: maximum redshift to which the abundance matching with the SDSS
        velocity dispersion function is valid
    :param cosmo: astropy.cosmology instance
    :type sky_area: `~astropy.units.Quantity`
    :param sky_area: Sky area over which galaxies are sampled. Must be in units of solid
        angle.
    :return: interpolation function f; f(stellar_mass) -> vel_disp
    """

    # selects galaxies with redshift below maximum redshift (z_max)
    bool_cut = galaxy_list["z"] < z_max
    galaxy_list_zmax = copy.deepcopy(galaxy_list[bool_cut])

    # number of selected galaxies
    num_select = len(galaxy_list_zmax)

    redshift = np.linspace(0, z_max, 100)
    z_list, vel_disp_list = vel_disp_sdss(
        sky_area, redshift, vd_min=50, vd_max=500, cosmology=cosmo, noise=True
    )

    # sort for stellar masses, largest values first
    galaxy_list_zmax.sort("stellar_mass", reverse=True)

    # sort velocity dispersion, largest values first
    vel_disp_list = np.flip(np.sort(vel_disp_list))
    num_vel_disp = len(vel_disp_list)
    # abundance match velocity dispersion with elliptical galaxy catalogue
    # abundance match velocity dispersion with elliptical galaxy catalogue
    if num_vel_disp >= num_select:
        galaxy_list_zmax["vel_disp"] = vel_disp_list[:num_select]
        # randomly select
    else:
        galaxy_list_zmax = galaxy_list_zmax[:num_vel_disp]
        galaxy_list_zmax["vel_disp"] = vel_disp_list
    # interpolate relationship between stellar mass and velocity dispersion
    stellar_mass = np.asarray(galaxy_list_zmax["stellar_mass"])
    vel_disp = np.asarray(galaxy_list_zmax["vel_disp"])

    # here we make sure we interpolate to low stellar masses
    stellar_mass = np.append(stellar_mass, 10**5)
    vel_disp = np.append(vel_disp, 10)
    # here we make sure we interpolate to high stellar mass
    max_stellar_mass = max(galaxy_list["stellar_mass"])
    max_vel_disp = vel_disp_from_m_star(max_stellar_mass)
    stellar_mass = np.append(max_stellar_mass, stellar_mass)
    vel_disp = np.append(max_vel_disp, vel_disp)
    f = interpolate.interp1d(
        x=np.log10(stellar_mass),
        y=vel_disp,
        fill_value=(0, max_vel_disp),
        bounds_error=False,
    )
    return f
