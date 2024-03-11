from scipy import integrate
from colossus.lss import mass_function
from colossus.cosmology import cosmology as colossus_cosmo
from hmf.cosmology.growth_factor import GrowthFactor
import numpy as np
import warnings
from astropy.units.quantity import Quantity


def colossus_halo_mass_function(m_200, cosmo, z, sigma8=0.81, ns=0.96):
    """M in Msun/h return dn/dlnM (mpc-3)"""

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
    colossus_cosmo.setCosmology(cosmo_name="halo_cosmo", **params)
    # todo: change cosmology with ...
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
    if np.array_equal(z, np.array([np.nan])):
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
    cosmology,
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
    wavenumber : array_like, optional
        The wave number array for power spectrum.
    collapse_function : function, optional
        The halo collapse function.
    power_spectrum : function, optional
        The power spectrum function.
    params : tuple, optional
        The parameters for the collapse function.

    Returns
    -------
    array
        An array of redshifts of Halos.
    """
    if cosmology is None:
        warnings.warn(
            "No cosmology provided, instead uses flat LCDM with default parameters"
        )
        from astropy.cosmology import FlatLambdaCDM

        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

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
    N = np.trapz(dN_dz, redshift_list)
    N = np.random.poisson(N)
    return N


def dndz_to_redshifts(N, dN_dz, redshift_list):
    # cumulative trapezoidal rule to get redshift CDF
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
    if z is np.array([np.nan]):
        return 0
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
    """Used in yml creating redshift list for mass sheet correction."""
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
