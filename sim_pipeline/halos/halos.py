from skypy.halos.mass import halo_mass_sampler
from skypy.halos.mass import halo_mass_function
from skypy.power_spectrum import eisenstein_hu
from scipy import integrate
from skypy.halos.mass import ellipsoidal_collapse_function
from skypy.halos.mass import press_schechter_collapse_function
from hmf.cosmology.growth_factor import GrowthFactor
import numpy as np
import warnings


def number_density_at_redshift(z, m_min=None, m_max=None, resolution=None, wavenumber=None, power_spectrum=None,
                               cosmology=None, collapse_function=None, params=None):
    """
    Function to calculate the cumulative number density of halos at a given redshift.

    :param z: The redshift at which to evaluate the number density.
    :type z: float
    :param m_min: The minimum halo mass.
    :type m_min: float, optional
    :param m_max: The maximum halo mass.
    :type m_max: float, optional
    :param resolution: The resolution of the mass grid.
    :type resolution: int, optional
    :param wavenumber: The wave number array for power spectrum.
    :type wavenumber: array, optional
    :param power_spectrum: The power spectrum function.
    :type power_spectrum: function, optional
    :param cosmology: The cosmology instance to use.
    :type cosmology: Cosmology, optional
    :param collapse_function: The halo collapse function.
    :type collapse_function: function, optional
    :param params: The parameters for the collapse function.
    :type params: tuple, optional
    :return: The cumulative number density of halos.
    :rtype: array
    """
    # define default parameters
    if m_min is None:
        m_min = 1e10
        warnings.warn("No minimum mass provided, instead uses 1e10 Msun")
        # TODO: make it only warn one time for one pipeline

    if m_max is None:
        m_max = 1e14
        warnings.warn("No maximum mass provided, instead uses 1e14 Msun")

    if resolution is None:
        resolution = 10000
        warnings.warn("No resolution provided, instead uses 10000")
    m = np.logspace(np.log10(m_min), np.log10(m_max), resolution)

    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses astropy.cosmology import default_cosmology")
        from astropy.cosmology import default_cosmology
        cosmology = default_cosmology.get()

    if wavenumber is None:
        wavenumber = np.logspace(-3, 1, num=resolution, base=10.0)
        warnings.warn("No wavenumber provided, instead uses logspace(-3, 1, num=resolution, base=10.0)")

    if collapse_function is None:
        collapse_function = ellipsoidal_collapse_function
        warnings.warn("No collapse function provided, instead uses ellipsoidal_collapse_function")

    if power_spectrum is None:
        power_spectrum = eisenstein_hu(wavenumber=wavenumber,cosmology=cosmology, A_s=2.1982e-09, n_s=0.969453)
        warnings.warn("No power spectrum provided, instead uses Eisenstein & Hu 1998")

    if params is None:
        params = (0.3, 0.7, 0.3, 1.686)
        warnings.warn("No collapse function parameters provided, instead uses (0.3, 0.7, 0.3, 1.686)")

    gf = GrowthFactor(cosmo=cosmology)
    growth_function = gf.growth_factor(z)

    massf = halo_mass_function(
        M=m, wavenumber=wavenumber, power_spectrum=power_spectrum, growth_function=growth_function,
        cosmology=cosmology, collapse_function=collapse_function, params=params)

    CDF = integrate.cumtrapz(massf, m, initial=0)
    return CDF


def growth_factor_at_redshift(z, cosmology=None):
    """
    Function to calculate the growth factor at a given redshift.

    :param z: The redshift at which to evaluate the growth factor.
    :type z: float
    :param cosmology: The cosmology instance to use.
    :type cosmology: Cosmology, optional
    :return: The growth factor at redshift z.
    :rtype: float
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
    Function to calculate an array of halo redshifts from a given comoving density.

    :param redshift_list: A list of redshifts.
    :type redshift_list: array
    :param sky_area: The area of the sky to consider (in square degrees).
    :type sky_area: float
    :param cosmology: The cosmology instance to use.
    :type cosmology: Cosmology, optional
    :param m_min: The minimum halo mass.
    :type m_min: float, optional
    :param m_max: The maximum halo mass.
    :type m_max: float, optional
    :param resolution: The resolution of the mass grid.
    :type resolution: int, optional
    :param wavenumber: The wavenumber array for power spectrum.
    :type wavenumber: array, optional
    :param collapse_function: The halo collapse function.
    :type collapse_function: function, optional
    :param power_spectrum: The power spectrum function.
    :type power_spectrum: function, optional
    :param params: The parameters for the collapse function.
    :type params: tuple, optional
    :return: An array of redshifts of halos.
    :rtype: array
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

    # integrate density to get expected number of galaxies
    N = np.trapz(dN_dz, redshift_list)
    N = int(N)

    # cumulative trapezoidal rule to get redshift CDF
    cdf = dN_dz  # reuse memory
    np.cumsum((dN_dz[1:] + dN_dz[:-1]) / 2 * np.diff(redshift_list), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]

    # sample N galaxy redshifts
    return np.interp(np.random.rand(N), cdf, redshift_list)


def halo_mass_at_z(z, m_min=None, m_max=None, resolution=None, wavenumber=None, power_spectrum=None, cosmology=None,
                   collapse_function=None, params=None):
    """
    Function to calculate the mass of halos at a given redshift.

    :param z: The redshift at which to evaluate the halo mass.
    :type z: float
    :param m_min: The minimum halo mass.
    :type m_min: float, optional
    :param m_max: The maximum halo mass.
    :type m_max: float, optional
    :param resolution: The resolution of the mass grid.
    :type resolution: int, optional
    :param wavenumber: The wavenumber array for power spectrum.
    :type wavenumber: array, optional
    :param power_spectrum: The power spectrum function.
    :type power_spectrum: function, optional
    :param cosmology: The cosmology instance to use.
    :type cosmology: Cosmology, optional
    :param collapse_function: The halo collapse function.
    :type collapse_function: function, optional
    :param params: The parameters for the collapse function.
    :type params: tuple, optional
    :return: The mass of halos at redshift z.
    :rtype: array
    """
    # TODO: debug
    if m_min is None:
        m_min = 1e10
        warnings.warn("No minimum mass provided, instead uses 1e10 Msun")
        # TODO: make it only warn one time for one pipeline

    if m_max is None:
        m_max = 1e14
        warnings.warn("No maximum mass provided, instead uses 1e14 Msun")


    if wavenumber is None:
        wavenumber = np.logspace(-3, 1, num=resolution, base=10.0)
        warnings.warn("No wavenumber provided, instead uses logspace(-3, 1, num=resolution, base=10.0)")

    if collapse_function is None:
        collapse_function = ellipsoidal_collapse_function
        warnings.warn("No collapse function provided, instead uses ellipsoidal collapse function")

    if params is None:
        params = (0.3, 0.7, 0.3, 1.686)
        warnings.warn("No collapse function parameters provided, instead uses (0.3, 0.7, 0.3, 1.686)")

    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses astropy.cosmology import default_cosmology")
        from astropy.cosmology import default_cosmology
        cosmology = default_cosmology.get()

    if power_spectrum is None:
        power_spectrum = eisenstein_hu(wavenumber=wavenumber,cosmology=cosmology, A_s=2.1982e-09, n_s=0.969453)
        warnings.warn("No power spectrum provided, instead uses Eisenstein & Hu 1998")

    if params is None:
        params = (0.3, 0.7, 0.3, 1.686)
        warnings.warn("No collapse function parameters provided, instead uses (0.3, 0.7, 0.3, 1.686)")
    try:
        iter(z)
    except TypeError:
        z = [z]

    mass = []
    for z_val in z:
        gf = GrowthFactor(cosmo=cosmology)
        growth_function = gf.growth_factor(z_val)

        mass.append(halo_mass_sampler(m_min=m_min, m_max=m_max, resolution=resolution, wavenumber=wavenumber,
                                      power_spectrum=power_spectrum, growth_function=growth_function, params=params,
                                      cosmology=cosmology, collapse_function=collapse_function))

    return mass
