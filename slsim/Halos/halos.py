from scipy import integrate
from colossus.lss import mass_function
from colossus.cosmology import cosmology as colossus_cosmo
from hmf.cosmology.growth_factor import GrowthFactor
import numpy as np
import warnings
from astropy.units.quantity import Quantity


def colossus_halo_mass_function(m_200, cosmo, z, sigma8=0.81, ns=0.96):
    """M in Msun/h return dn/dlnM (mpc-3) Calculates the differential halo mass function
    per logarithmic interval in mass at a given redshift.

    This function uses the Colossus library to calculate the halo mass function, given a mass scale (m_200),
    a cosmology (cosmo), and a redshift (z). The mass function is defined as dn/dlnM, where n is the number
    density of halos and M is the halo mass. The function allows for specification of the sigma8 and ns.

    Parameters
    ----------
    m_200 : ndarray
        Halo mass scale in units of solar mass divided by the Hubble parameter (M_sun/h).
    cosmo : astropy.cosmology instance
        The cosmology instance specifying the cosmological parameters.
    z : float
        Redshift at which the mass function is evaluated.
    sigma8 : float, optional
        Defaults to 0.81.
    ns : float, optional
        Defaults to 0.96.

    Returns
    -------
    ndarray
        The differential halo mass function dn/dlnM in units of Mpc^-3.

    Notes
    -----
    The 'bhattacharya11' model is used for the mass function within the Colossus framework.
    """
    params = dict(
        flat=(cosmo.Ok0 == 0.0),
        H0=cosmo.H0.value,
        Om0=cosmo.Om0,
        Ode0=cosmo.Ode0,
        Ob0=cosmo.Ob0,
        Tcmb0=cosmo.Tcmb0.value,
        Neff=cosmo.Neff,
        sigma8=sigma8,
        ns=ns,
    )
    # TODO: seems like still not working for other cosmology rather than default
    colossus_cosmo.setCosmology(cosmo_name="halo_cosmo", **params)
    h3 = np.power(cosmo.h, 3)
    mfunc_h3_dmpc3 = mass_function.massFunction(
        m_200, z, mdef="fof", model="bhattacharya11", q_out="dndlnM"
    )
    # in h^3*Mpc-3
    massf = mfunc_h3_dmpc3 * h3  # in Mpc-3
    return massf


def get_value_if_quantity(variable):
    if isinstance(variable, Quantity):
        return variable.value
    else:
        return variable


def colossus_halo_mass_sampler(
    m_min,
    m_max,
    resolution,
    z,
    cosmology,
    sigma8=0.81,
    ns=0.96,
    size=None,
):
    """Samples halo masses from a mass function within specified minimum and maximum
    masses.

    This function computes a halo mass function using the Colossus library, then samples
    halo masses from this distribution. The mass function is computed within a given mass range
    and at a specific redshift, using a specified cosmology and optional sigma8 and ns parameters
    for the power spectrum normalization and spectral index, respectively.

    Parameters
    ----------
    m_min : Quantity or float
        The minimum halo mass (in M_sol/h). If an astropy Quantity, will be converted to its value.
    m_max : Quantity or float
        The maximum halo mass (in M_sol/h). If an astropy Quantity, will be converted to its value.
    resolution : Quantity or int
        The number of mass bins to use for the mass function calculation. If an astropy Quantity, will be converted to its value.
    z : float
        The redshift at which to compute the halo mass function.
    cosmology : astropy.cosmology instance
        The cosmology instance to use for calculating the halo mass function.
    sigma8 : float, optional
        Sigma8 parameter. Default is 0.81.
    ns : float, optional
        ns in colossus cosmo setting. Default is 0.96.
    size : int or None, optional
        The number of random samples to draw. If None, a single value is returned.

    Returns
    -------
    ndarray or float
        The sampled halo masses (in M_sol), with the number of samples determined by the `size` parameter.
        If `size` is None, a single float value is returned.

    Notes
    -----
    The halo mass function is computed using the Colossus library, specifically using the
    'bhattacharya11' model for the mass function. The cosmology parameters used in the calculation
    are extracted from the provided cosmology instance and used to set a temporary cosmology
    in Colossus with the name "halo_cosmo".
    """
    m_min_value = get_value_if_quantity(m_min)
    m_max_value = get_value_if_quantity(m_max)
    h_value = get_value_if_quantity(cosmology.h)
    resolution_value = get_value_if_quantity(resolution)

    minh = m_min_value * h_value
    maxh = m_max_value * h_value

    m = np.geomspace(minh, maxh, resolution_value)
    massf = colossus_halo_mass_function(
        m,
        cosmology,
        z,
        sigma8,
        ns,
    )

    CDF = integrate.cumtrapz(massf, np.log(m), initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size=size)
    return np.interp(n_uniform, CDF, m) / cosmology.h


def set_defaults(
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
):
    """Utility function to set default values for parameters if not provided.

    Parameters
    ----------
    m_min : float, optional
        The minimum halo mass (in M_sol).
    m_max : float, optional
        The maximum halo mass (in M_sol).
    resolution : int, optional
        The resolution of the grid.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.

    Returns
    -------
    tuple
        The parameters with default values set where applicable.
    """

    # Default values
    if m_min is None:
        m_min = 1.0e12
        warnings.warn("No minimum mass provided, instead uses 1e10 Msun")

    if m_max is None:
        m_max = 1.0e14
        warnings.warn("No maximum mass provided, instead uses 1e14 Msun")

    if resolution is None:
        resolution = 100
        warnings.warn("No resolution provided, instead uses 100")

    if cosmology is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology import default_cosmology"
        )
        from astropy.cosmology import default_cosmology

        cosmology = default_cosmology.get()

    return (
        m_min,
        m_max,
        resolution,
        cosmology,
    )


def number_density_at_redshift(
    z,
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
    sigma8=0.81,
    ns=0.96,
):
    """Calculates the number density of halos at specified redshifts within a given mass
    range.

    This function computes the number density of halos per comoving volume unit at different redshifts,
    considering halos within a specified mass range. The number density is obtained by integrating the
    halo mass function over the mass range.

    Parameters
    ----------
    z : float or ndarray
        Redshift or an array of redshifts at which to calculate the number density.
    m_min : float, optional
        The minimum mass of halos to include in the calculation (in M_sol).
    m_max : float, optional
        The maximum mass of halos to include in the calculation (in M_sol).
    resolution : int, optional
        The number of mass bins to use for the mass function calculation.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use for calculating the halo mass function.
    sigma8 : float, optional
        Default is 0.81.
    ns : float, optional
        Default is 0.96.

    Returns
    -------
    list or ndarray
        The number density of halos (in Mpc^-3) at each specified redshift. The output is a list if z is
        a scalar and an ndarray if z is an array.

    Notes
    -----
    If the input redshift(s) contain NaN values, a warning is issued, and a default redshift of 0.0001 is used.
    The function integrates the halo mass function, obtained from the `colossus_halo_mass_function` function,
    over the specified mass range to calculate the total number density.
    """
    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults(
        m_min,
        m_max,
        resolution,
        cosmology,
    )
    m_200 = np.geomspace(m_min * cosmology.h, m_max * cosmology.h, resolution)
    if np.all(np.isnan(z)):
        warnings.warn("Redshift data lost, instead uses 0.0001")
        return [0.0001] * len(z)
    if isinstance(z, float):
        z = np.array([z])
    cdfs = []
    for zi in z:
        massf = colossus_halo_mass_function(
            m_200,
            cosmology,
            zi,
            sigma8=sigma8,
            ns=ns,
        )
        total_number_density = number_for_certain_mass(massf, m_200, dndlnM=True)
        cdfs.append(total_number_density)
    return cdfs


def number_for_certain_mass(massf, m, dndlnM=False):
    # massf:Mpc-3 Msun-1
    # output: number per Mpc3 at certain redshift
    if dndlnM:
        # massf: dm/dlnM200 Mpc-3
        # output: number per Mpc3 at certain redshift
        return integrate.trapz(massf, np.log(m))
    else:
        return integrate.trapz(massf * m, np.log(m))


def growth_factor_at_redshift(z, cosmology):
    """Calculate the growth factor at a given redshift.

    Parameters
    ----------
    z : float, array_like, or list
        The redshift at which to evaluate the growth factor.
    cosmology : astropy.cosmology instance
        The cosmology instance to use.

    Returns
    -------
    float or numpy.ndarray
        The growth factor at redshift z.

    Notes
    -----
    Using hmf library to calculate growth factor.
    """
    # Check if z is a list and convert to numpy array if it is
    if isinstance(z, list):
        z = np.array(z)

    gf = GrowthFactor(cosmo=cosmology)
    growth_function = gf.growth_factor(z)
    if not isinstance(growth_function, (list, np.ndarray)):
        growth_function = [growth_function]
    return growth_function


def redshift_halos_array_from_comoving_density(
    redshift_list,
    sky_area,
    cosmology=None,
    m_min=None,
    m_max=None,
    resolution=None,
):
    """Calculate an array of halo redshifts from a given comoving density.

    The function computes the expected number of Halos at different
    redshifts using the differential comoving volume
    and the halo number density then apply the poisson distribution on it.
    After determining the expected number of Halos, the function produces
    an array of halo redshifts based on the cumulative distribution
    function (CDF) of the number density.

    Parameters
    ----------
    redshift_list : array_like
        A list of redshifts.
    sky_area : `~astropy.units.Quantity`
        The area of the sky to consider (in square degrees).
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.
    m_min : float, optional
        The minimum halo mass (in M_sol).
    m_max : float, optional
        The maximum halo mass (in M_sol).
    resolution : int, optional
        The resolution of the mass grid.

    Returns
    -------
    array
        An array of redshifts of Halos.
    """
    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses default ")
        from astropy.cosmology import default_cosmology

        cosmology = default_cosmology.get()

    dV_dz = v_per_redshift(redshift_list, cosmology, sky_area)
    # dV_dz is in "Mpc3"

    dN_dz = dv_dz_to_dn_dz(
        dV_dz,
        redshift_list,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
    )

    # integrate density to get expected number of Halos
    N = dndz_to_N(dN_dz, redshift_list)
    if N == 0:
        warnings.warn("No Halos found in the given redshift range")
        return np.array([np.nan])
    else:
        return dndz_to_redshifts(N, dN_dz, redshift_list)


def dv_dz_to_dn_dz(
    dV_dz,
    redshift_list,
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
):
    """Converts a differential comoving volume element dV/dz to a differential number
    density dn/dz.

    This function calculates the number density of objects per redshift interval by multiplying the
    differential comoving volume element dV/dz by the number density of objects per comoving volume
    at each redshift.

    Parameters
    ----------
    dV_dz : ndarray
        Differential comoving volume element as a function of redshift.
    redshift_list : ndarray
        List of redshifts corresponding to dV_dz values.
    m_min : float, optional
        The minimum mass of halos to consider in the number density calculation.
    m_max : float, optional
        The maximum mass of halos to consider in the number density calculation.
    resolution : int, optional
        The resolution of the grid used in the mass function calculation.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance used for the calculation.

    Returns
    -------
    ndarray
        The differential number density dn/dz, which represents the number of objects per redshift
        interval per comoving volume unit.

    Notes
    -----
    The function requires the specification of halo mass range (m_min, m_max) and cosmology to compute
    the number density of objects. These parameters are optional and can be set to default values if not provided.
    """
    density = number_density_at_redshift(
        z=redshift_list,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
    )  # dn/dv at z; Mpc-3
    dV_dz *= density
    return dV_dz


def dndz_to_N(dN_dz, redshift_list):
    """Converts a redshift distribution dN/dz into a total number of objects N using
    Poisson statistics.

    Parameters
    ----------
    dN_dz : ndarray
        The differential number of objects per redshift interval.
    redshift_list : ndarray
        List of redshifts corresponding to the dN_dz values.

    Returns
    -------
    int
        The total number of objects, drawn from a Poisson distribution with the mean set to the integral
        of dN/dz over the redshift range.
    """
    N = np.trapz(dN_dz, redshift_list)
    N = np.random.poisson(N)
    return N


def dndz_to_redshifts(N, dN_dz, redshift_list):
    """Cumulative trapezoidal rule to get redshift CDF.

    Parameters
    ----------
    N : int
        The number of redshifts to sample.
    dN_dz : ndarray
        The differential number of objects per redshift interval.
    redshift_list : ndarray
        List of redshifts corresponding to the dN_dz values.

    Returns
    -------
    ndarray
        An array of N sampled redshifts based on the provided distribution.
    """
    assert len(dN_dz) == len(redshift_list)
    cdf = dN_dz  # reuse memory
    np.cumsum((dN_dz[1:] + dN_dz[:-1]) / 2 * np.diff(redshift_list), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]
    return np.interp(np.random.rand(N), cdf, redshift_list)


def v_per_redshift(redshift_list, cosmology, sky_area):
    """Calculate the volume per redshift.

    Parameters
    ----------
    redshift_list : array_like
        A list of redshifts.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.
    sky_area : `~astropy.units.Quantity`
        The area of the sky to consider (in square degrees) or solid angle.

    Returns
    -------
    array
        The volume per redshift in Mpc3.
    """
    dV_dz = (cosmology.differential_comoving_volume(redshift_list) * sky_area).to_value(
        "Mpc3"
    )
    return dV_dz


def halo_mass_at_z(
    z,
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
    sigma8=0.81,
    ns=0.96,
):
    """"""

    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults(
        m_min,
        m_max,
        resolution,
        cosmology,
    )
    try:
        iter(z)
    except TypeError:
        z = [z]

    mass = []
    if np.all(np.isnan(z)):
        return [0] * len(z)
    for z_val in z:
        mass.append(
            colossus_halo_mass_sampler(
                m_min=m_min,
                m_max=m_max,
                resolution=resolution,
                z=z_val,
                cosmology=cosmology,
                size=1,
                sigma8=sigma8,
                ns=ns,
            )
        )
    return mass


def redshift_mass_sheet_correction_array_from_comoving_density(redshift_list):
    """Generates an array of redshift values for use in mass sheet correction
    calculations.

    This function creates an array of redshift values starting from 0.025 up to the maximum redshift in the
    provided redshift_list, with a step of 0.05.

    Parameters
    ----------
    redshift_list : ndarray
        An array of redshift values, where the maximum value is used to set the range of the output array.

    Returns
    -------
    ndarray
        An array of redshift values starting from 0.025 up to the maximum redshift in redshift_list, with a step of 0.05.
    """
    z_max = redshift_list[-1]
    linspace_values = np.arange(0.025, z_max, 0.05)
    return linspace_values


def determinism_kappa_first_moment_at_redshift(z):
    """
    m_min= 1e12
    m_max= 1e16
    z=5
    Intercept: 0.0002390784813232419
    Coefficients:
        Degree 0: 0.0
        Degree 1: -0.0014658189854554395
        Degree 2: -0.11408175546088226
        Degree 3: 0.1858161514337054
        Degree 4: -0.14580188720668946
        Degree 5: 0.07179490182290658
        Degree 6: -0.023842218143709567
        Degree 7: 0.00534416068166958
        Degree 8: -0.0007728539951923031
        Degree 9: 6.484537448337964e-05
        Degree 10: -2.389378848385584e-06
    Parameters
    ----------
    z

    Returns
    -------
    """
    # todo: almost deprecated, considering change to another mass
    # sheet correction method
    m_ls = []
    for zi in z:
        m2 = (
            0.0002390784813232419
            + -0.0014658189854554395 * zi
            + -0.11408175546088226 * (zi**2)
            + 0.1858161514337054 * (zi**3)
            + -0.14580188720668946 * (zi**4)
            + 0.07179490182290658 * (zi**5)
            + -0.023842218143709567 * (zi**6)
            + 0.00534416068166958 * (zi**7)
            + -0.0007728539951923031 * (zi**8)
            + 6.484537448337964e-05 * (zi**9)
            + -2.389378848385584e-06 * (zi**10)
        )
        m_ls.append(m2)
    return m_ls
