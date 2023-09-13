from sim_pipeline.Skypy_halos_duplicate.halos.mass import halo_mass_sampler
from sim_pipeline.Skypy_halos_duplicate.halos.mass import halo_mass_function
from sim_pipeline.Skypy_halos_duplicate.power_spectrum import eisenstein_hu
from scipy import integrate
from sim_pipeline.Skypy_halos_duplicate.halos.mass import ellipsoidal_collapse_function
from sim_pipeline.Skypy_halos_duplicate.halos.mass import press_schechter_collapse_function
from hmf.cosmology.growth_factor import GrowthFactor
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import numpy as np
import warnings


def set_defaults(m_min=None, m_max=None, wavenumber=None, resolution=None, power_spectrum=None, cosmology=None,
                 collapse_function=None, params=None):
    """
    Utility function to set default values for parameters if not provided.

    Parameters
    ----------
    m_min : float, optional
        The minimum halo mass (in M_sol).
    m_max : float, optional
        The maximum halo mass (in M_sol).
    wavenumber : array_like, optional
        The wavenumber array for power spectrum.
    resolution : int, optional
        The resolution of the grid.
    power_spectrum : function, optional
        The power spectrum function.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.
    collapse_function : function, optional
        The halo collapse function.
    params : tuple, optional
        The parameters for the collapse function.

    Returns
    -------
    tuple
        The parameters with default values set where applicable.

    """

    # Default values
    if m_min is None:
        m_min = 1E+10
        warnings.warn("No minimum mass provided, instead uses 1e10 Msun")

    if m_max is None:
        m_max = 1E+14
        warnings.warn("No maximum mass provided, instead uses 1e14 Msun")

    if resolution is None:
        resolution = 1000
        warnings.warn("No resolution provided, instead uses 10000")

    if wavenumber is None:
        wavenumber = np.logspace(-3, 1, num=resolution, base=10.0)
        warnings.warn("No wavenumber provided, instead uses logspace(-3, 1, num=resolution, base=10.0)")

    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses astropy.cosmology import default_cosmology")
        from astropy.cosmology import default_cosmology
        cosmology = default_cosmology.get()

    if collapse_function is None:
        collapse_function = ellipsoidal_collapse_function
        warnings.warn("No collapse function provided, instead uses ellipsoidal_collapse_function")

    if power_spectrum is None:
        power_spectrum = eisenstein_hu(wavenumber=wavenumber, cosmology=cosmology, A_s=2.1982e-09, n_s=0.969453)
        warnings.warn("No power spectrum provided, instead uses Eisenstein & Hu 1998")

    if params is None:
        params = (0.3, 0.7, 0.3, 1.686)
        warnings.warn("No collapse function parameters provided, instead uses (0.3, 0.7, 0.3, 1.686)")

    return m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params


def number_density_at_redshift(z, m_min=None, m_max=None, resolution=None, wavenumber=None, power_spectrum=None,
                               cosmology=None, collapse_function=None, params=None):
    """
    Function to calculate the cumulative number density of Halos at a given redshift.

    Parameters
    ----------
    z : float or array_like
        The redshift at which to evaluate the number density.
    m_min : float, optional
        The minimum halo mass (in M_sol).
    m_max : float, optional
        The maximum halo mass (in M_sol).
    resolution : int, optional
        The resolution of the mass grid.
    wavenumber : array_like, optional
        The wave number array for power spectrum.
    power_spectrum : function, optional
        The power spectrum function.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.
    collapse_function : function, optional
        The halo collapse function.
    params : tuple, optional
        The parameters for the collapse function.

    Returns
    -------
    array
        The cumulative number density of Halos.

    """
    # define default parameters
    m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params = set_defaults(
        m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params)

    m = np.logspace(np.log10(m_min), np.log10(m_max), resolution)

    gf = GrowthFactor(cosmo=cosmology)
    if z is np.array([np.nan]):
        return 0
    else:
        growth_function = gf.growth_factor(z)

        massf = halo_mass_function(
            M=m, wavenumber=wavenumber, power_spectrum=power_spectrum, growth_function=growth_function,
            cosmology=cosmology, collapse_function=collapse_function, params=params)

        CDF = integrate.cumtrapz(massf, m, initial=0)
        return CDF


def growth_factor_at_redshift(z, cosmology=None):
    """
    Calculate the growth factor at a given redshift.

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the growth factor.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.

    Returns
    -------
    float
        The growth factor at redshift z.

    Notes
    -----
         Using hmf library to calculate growth factor.

    """
    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
        from astropy.cosmology import FlatLambdaCDM
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    gf = GrowthFactor(cosmo=cosmology)
    growth_function = gf.growth_factor(z)
    return growth_function


def redshift_halos_array_from_comoving_density(redshift_list, sky_area, cosmology, m_min=None, m_max=None,
                                               resolution=None, wavenumber=None, collapse_function=None,
                                               power_spectrum=None, params=None):
    """
    Calculate an array of halo redshifts from a given comoving density.

    The function computes the expected number of Halos at different redshifts using the differential comoving volume
    and the halo number density then apply the poisson distribution on it. After determining the expected number of
    Halos, the function produces an array of halo redshifts based on the cumulative distribution function (CDF) of
    the number density.

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
        warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
        from astropy.cosmology import FlatLambdaCDM
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    dN_dz = (cosmology.differential_comoving_volume(redshift_list) * sky_area).to_value('Mpc3')
    density = number_density_at_redshift(z=redshift_list, m_min=m_min, m_max=m_max, resolution=resolution,
                                         wavenumber=wavenumber,
                                         power_spectrum=power_spectrum, cosmology=cosmology,
                                         collapse_function=collapse_function, params=params)
    dN_dz *= density

    # integrate density to get expected number of Halos
    N = np.trapz(dN_dz, redshift_list)
    N_0 = int(N)
    N = np.random.poisson(N_0)
    if N == 0:
        warnings.warn("No Halos found in the given redshift range")
        return np.array([np.nan])
    else:
        # cumulative trapezoidal rule to get redshift CDF
        cdf = dN_dz  # reuse memory
        np.cumsum((dN_dz[1:] + dN_dz[:-1]) / 2 * np.diff(redshift_list), out=cdf[1:])
        cdf[0] = 0
        cdf /= cdf[-1]
        return np.interp(np.random.rand(N), cdf, redshift_list)


def halo_mass_at_z(z, m_min=None, m_max=None, resolution=None, wavenumber=None, power_spectrum=None, cosmology=None,
                   collapse_function=None, params=None):
    """
    Calculate the mass of Halos at a given redshift (list).

    Parameters
    ----------
    z : float or array_like
        The redshift at which to evaluate the halo mass.
    m_min : float, optional
        The minimum halo mass (in M_sol).
    m_max : float, optional
        The maximum halo mass (in M_sol).
    resolution : int, optional
        The resolution of the mass grid.
    wavenumber : array_like, optional
        The wavenumber array for power spectrum.
    power_spectrum : function, optional
        The power spectrum function.
    cosmology : astropy.cosmology instance, optional
        The cosmology instance to use.
    collapse_function : function, optional
        The halo collapse function.
    params : tuple, optional
        The parameters for the collapse function.

    Returns
    -------
    array
        The mass of Halos at redshift z.

    """

    m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params = set_defaults(
        m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params)
    try:
        iter(z)
    except TypeError:
        z = [z]

    mass = []
    if z is np.array([np.nan]):
        return 0
    for z_val in z:
        gf = GrowthFactor(cosmo=cosmology)
        growth_function = gf.growth_factor(z_val)

        mass.append(halo_mass_sampler(m_min=m_min, m_max=m_max, resolution=resolution, wavenumber=wavenumber,
                                      power_spectrum=power_spectrum, growth_function=growth_function, params=params,
                                      cosmology=cosmology, collapse_function=collapse_function, size=1))

    return mass


def mass_first_moment_at_redshift(z, m_min=None, m_max=None, resolution=None, wavenumber=None, power_spectrum=None,
                                  cosmology=None, collapse_function=None, params=None):
    # define default parameters
    m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params = set_defaults(
        m_min, m_max, wavenumber, resolution, power_spectrum, cosmology, collapse_function, params)

    m = np.logspace(np.log10(m_min), np.log10(m_max), resolution)
    expectation_m_result = []
    gf = GrowthFactor(cosmo=cosmology)
    if z is np.array([np.nan]):
        return 0
    else:
        for z_val in z:
            growth_function = gf.growth_factor(z_val)

            massf = halo_mass_function(
                m, wavenumber, power_spectrum, growth_function,
                cosmology, collapse_function, params=params)

            CDF = integrate.cumtrapz(massf, m, initial=0)
            CDF = CDF / CDF[-1]
            expectation_m_result.append(np.interp(0.5, CDF, m))
        return expectation_m_result


def redshift_mass_sheet_correction_array_from_comoving_density(redshift_list, sky_area, cosmology, m_min=None,
                                                               m_max=None,
                                                               resolution=None, wavenumber=None, collapse_function=None,
                                                               power_spectrum=None, params=None):
    """

    """
    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
        from astropy.cosmology import FlatLambdaCDM
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    dN_dz = (cosmology.differential_comoving_volume(redshift_list) * sky_area).to_value('Mpc3')
    density = number_density_at_redshift(z=redshift_list, m_min=m_min, m_max=m_max, resolution=resolution,
                                         wavenumber=wavenumber,
                                         power_spectrum=power_spectrum, cosmology=cosmology,
                                         collapse_function=collapse_function, params=params)
    dN_dz *= density

    # integrate density to get expected number of Halos
    N = np.trapz(dN_dz, redshift_list)
    N_0 = int(N)
    if N_0 == 0:
        warnings.warn("No Halos found in the given redshift range")
        return np.array([np.nan])
    else:
        cdf = dN_dz  # reuse memory
        np.cumsum((dN_dz[1:] + dN_dz[:-1]) / 2 * np.diff(redshift_list), out=cdf[1:])
        cdf[0] = 0
        cdf /= cdf[-1]
        return np.interp(np.linspace(1 / N_0, 1, N_0), cdf, redshift_list)


def deg2_to_cone_angle(solid_angle_deg2):
    """
    Convert solid angle in square degrees to half cone angle in radians.

    Parameters
    ----------
    solid_angle_deg2 : float
        The solid angle in square degrees to be converted.

    Returns
    -------
    float
        The cone angle in radians corresponding to the provided solid angle.

    Notes
    -----
    This function calculates the cone angle using the relationship between
    the solid angle in steradians and the cone's apex angle.
    """
    solid_angle_sr = solid_angle_deg2 * (np.pi / 180) ** 2
    theta = np.arccos(1 - solid_angle_sr / (2 * np.pi))  # rad
    return theta


def cone_radius_angle_to_physical_area(radius_rad, z, cosmo):
    """
    Convert cone radius angle to physical area at a given redshift.

    Parameters
    ----------
    radius_rad : float
        The half cone's angle in radians.
    z : float
        The redshift at which the physical area is to be calculated.
    cosmo : astropy.Cosmology instance
        The cosmology used for the conversion.

    Returns
    -------
    float
        The physical area in Mpc^2 corresponding to the given cone radius and redshift.

    Notes
    -----
    The function calculates the physical area of a patch of the sky with
    a specified cone angle and redshift using the angular diameter distance.
    """
    physical_radius = cosmo.angular_diameter_distance(z) * radius_rad  # Mpc
    area_physical = np.pi * physical_radius ** 2
    return area_physical  # in Mpc2


def kappa_ext_for_each_sheet(redshift_list, first_moment, sky_area, cosmology):
    cone_opening_angle = deg2_to_cone_angle(sky_area.value)
    # TODO: make it possible for other geometry model

    lens_cosmo = LensCosmo(z_lens=redshift_list,
                           z_source=10, cosmo=cosmology)
    epsilon_crit = lens_cosmo.sigma_crit

    area = cone_radius_angle_to_physical_area(cone_opening_angle, redshift_list, cosmology)  # mpc2
    first_moment_d_area = np.divide(np.array(first_moment), np.array(area))
    kappa_ext = np.divide(first_moment_d_area, epsilon_crit)
    return -kappa_ext
