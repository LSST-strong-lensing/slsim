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
        warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
        from astropy.cosmology import FlatLambdaCDM
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    if wavenumber is None:
        wavenumber = np.logspace(-3, 1, num=resolution, base=10.0)
        warnings.warn("No wavenumber provided, instead uses logspace(-3, 1, num=resolution, base=10.0)")

    if collapse_function is None:
        collapse_function = press_schechter_mass_function
        warnings.warn("No collapse function provided, instead uses press_schechter_mass_function")

    if power_spectrum is None:
        power_spectrum = eisenstein_hu
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
    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
        from astropy.cosmology import FlatLambdaCDM
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    dN_dz = (cosmology.differential_comoving_volume(redshift_list) * sky_area).to_value('Mpc3')
    density = number_density_at_redshift(z= redshift_list, m_min= m_min, m_max=m_max, resolution=resolution,
                                         wavenumber=wavenumber,
                                         power_spectrum = power_spectrum, cosmology=cosmology,
                                         collapse_function= collapse_function, params=params)
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
    #TODO: debug
    if m_min is None:
        m_min = 1e10
        warnings.warn("No minimum mass provided, instead uses 1e10 Msun")

    if m_max is None:
        m_max = 1e14
        warnings.warn("No maximum mass provided, instead uses 1e14 Msun")

    if resolution is None:
        resolution = 10000
        warnings.warn("No resolution provided, instead uses 10000")

    if wavenumber is None:
        wavenumber = np.logspace(-3, 1, num=resolution, base=10.0)
        warnings.warn("No wavenumber provided, instead uses logspace(-3, 1, num=resolution, base=10.0)")

    if collapse_function is None:
        collapse_function = press_schechter_mass_function
        warnings.warn("No collapse function provided, instead uses ellipsoidal collapse function")

    if params is None:
        params = (0.3, 0.7, 0.3, 1.686)
        warnings.warn("No collapse function parameters provided, instead uses (0.3, 0.7, 0.3, 1.686)")

    if power_spectrum is None:
        power_spectrum = eisenstein_hu
        warnings.warn("No power spectrum provided, instead uses Eisenstein & Hu 1998")

    if cosmology is None:
        warnings.warn("No cosmology provided, instead uses flat LCDM with default parameters")
        from astropy.cosmology import FlatLambdaCDM
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

    gf = GrowthFactor(cosmo=cosmology)
    growth_function = gf.growth_factor(z)

    mass = halo_mass_sampler(m_min=m_min, m_max=m_max, resolution=resolution, wavenumber=wavenumber,
                             power_spectrum=power_spectrum, growth_function=growth_function, params=params,
                             cosmology=cosmology, collapse_function=collapse_function)

    return mass
