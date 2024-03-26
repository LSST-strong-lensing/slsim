from scipy import integrate
from colossus.lss import mass_function
from colossus.cosmology import cosmology as colossus_cosmo
from hmf.cosmology.growth_factor import GrowthFactor
import numpy as np
import warnings
from astropy.units.quantity import Quantity


def colossus_halo_mass_function(m_200, cosmo, z, sigma8=0.81, ns=0.96, omega_m=None):
    """Calculate the differential halo mass function per logarithmic mass interval at a
    given redshift.

    This function leverages the Colossus library to determine the halo mass function
    based on a mass scale (m_200), a cosmological model (cosmo), and a redshift (z). The
    mass function, expressed as dn/dlnM, represents the number density of halos per log
    mass interval. Parameters sigma8 and ns can be specified to adjust the calculation.

    :param m_200: Halo mass scale, in solar mass units divided by the Hubble parameter
        (M_sun/h).
    :param cosmo: Cosmology instance detailing the cosmological parameters.
    :param z: Redshift for evaluating the mass function.
    :param sigma8: Normalization of the power spectrum, defaults to 0.81 if not
        specified.
    :param ns: Spectral index, defaults to 0.96 if not specified.
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same
        in Cosmology setting.
    :type m_200: ndarray
    :type cosmo: astropy.cosmology instance
    :type z: float
    :type sigma8: float, optional
    :type ns: float, optional
    :type omega_m: float, optional
    :return: The differential halo mass function dn/dlnM, in units of Mpc^-3.
    :rtype: ndarray :note: The 'bhattacharya11' model within the Colossus framework is
        used for the mass function.
    """
    if omega_m is None:
        omega_m = cosmo.Om0
    elif isinstance(omega_m, float) or isinstance(omega_m, int):
        omega_m = omega_m
    else:
        print(omega_m, type(omega_m))
        raise ValueError("omega_m should be a float or int or None")
    params = dict(
        flat=(cosmo.Ok0 == 0.0),
        H0=cosmo.H0.value,
        Om0=omega_m,
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
    """Extracts the numerical value from an astropy Quantity object or returns the input
    if not a Quantity.

    This function checks if the input variable is an instance of an astropy Quantity. If
    it is, the function extracts and returns the numerical value of the Quantity. If the
    input is not a Quantity, it returns the input variable unchanged.

    :param variable: The variable to be checked and possibly converted. Can be an
        astropy Quantity or any other data type.
    :type variable: Quantity or any
    :return: The numerical value of the Quantity if the input is a Quantity; otherwise,
        the input variable itself.
    :rtype: float or any
    """
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
    omega_m=None,
):
    """Sample halo masses from a mass function within defined mass boundaries.

    Utilizes the Colossus library to compute a halo mass function, from which halo masses are sampled. The function calculates the mass function across a specified mass range and redshift, using given cosmological parameters and, optionally, values for sigma8 and ns for power spectrum normalization and spectral index, respectively.

    :param m_min: Minimum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param m_max: Maximum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param resolution: Number of mass bins for the mass function calculation, converted to value if specified as an astropy Quantity.
    :param z: Redshift for halo mass function computation.
    :param cosmology: Cosmology instance for calculating the halo mass function.
    :param sigma8: Sigma8 parameter, default is 0.81.
    :param ns: Spectral index in Colossus cosmology settings, default is 0.96.
    :param size: Number of random samples to draw. Returns a single value if None.
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same in Cosmology setting.
    :type m_min: Quantity or float
    :type m_max: Quantity or float
    :type resolution: Quantity or int
    :type z: float
    :type cosmology: astropy.cosmology instance
    :type sigma8: float, optional
    :type ns: float, optional
    :type size: int or None, optional
    :type omega_m: float
    :return: Sampled halo masses in M_sol, the number of samples is determined by the `size` parameter. Returns a single float if `size` is None.
    :rtype: ndarray or float
    :note: The Colossus library's 'bhattacharya11' model is used for the mass function. Cosmology parameters from the provided instance are temporarily applied to Colossus as "halo_cosmo".
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
        omega_m,
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
    """Set default values for parameters if not explicitly provided.

    Ensure all parameters have values, setting defaults for any that are unspecified.
    This utility function aids in initializing simulation or calculation setups where
    certain parameters might not be critical and can assume default values.

    :param m_min: The minimum halo mass, in solar mass units (M_sol). Defaults to a
        predetermined value if not specified.
    :param m_max: The maximum halo mass, in solar mass units (M_sol). Defaults to a
        predetermined value if not specified.
    :param resolution: Resolution of the computational grid. Assigned a default value if
        omitted.
    :param cosmology: Cosmology model instance to be used in calculations. If not
        provided, a default cosmology is used.
    :type m_min: float, optional
    :type m_max: float, optional
    :type resolution: int, optional
    :type cosmology: astropy.cosmology instance, optional
    :return: A tuple containing the parameters, with defaults applied where necessary.
    :rtype: tuple
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
    omega_m=None,
):
    """Compute the number density of halos at various redshifts within a specific mass
    range.

    This function integrates the halo mass function over a given mass range to find the
    number density of halos per unit comoving volume at different redshifts. It's useful
    for understanding the distribution and evolution of halos across cosmic time.

    :param z: Redshift(s) at which to compute the number density, can be a single value
        or an array.
    :param m_min: Minimum halo mass included in the density calculation, in solar masses
        (M_sol). Optional, with a default value if not provided.
    :param m_max: Maximum halo mass for the density calculation, in solar masses
        (M_sol). Optional, with a default value if not provided.
    :param resolution: Number of mass bins for integrating the mass function. Optional,
        defaults to a predetermined value.
    :param cosmology: Cosmology instance for the underlying cosmological model.
        Optional, defaults to a standard model if not provided.
    :param sigma8: Normalization of the power spectrum, optional, with a default value
        if not specified.
    :param ns: Spectral index for the power spectrum, optional, with a default value if
        not specified.
    :param omega_m: Matter density parameter, optional, will use the cosmology setting
        if not specified.
    :type z: float or ndarray
    :type m_min: float, optional
    :type m_max: float, optional
    :type resolution: int, optional
    :type cosmology: astropy.cosmology instance, optional
    :type sigma8: float, optional
    :type ns: float, optional
    :type omega_m: float, optional
    :return: Number density of halos at each specified redshift, as a list (for scalar
        z) or ndarray (for array z).
    :rtype: list or ndarray :note: A warning is issued for NaN values in input
        redshifts, with a fallback to a default redshift of 0.0001.
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
            m_200, cosmology, zi, sigma8=sigma8, ns=ns, omega_m=omega_m
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
    """Determine the growth factor at specified redshift(s).

    Calculates the growth of structure over cosmic time by evaluating the growth factor
    at given redshift(s). This is crucial for understanding how density fluctuations
    evolve in the universe.

    :param z: Redshift(s) at which to evaluate the growth factor, can be a single value
        or an array.
    :param cosmology: Cosmology instance dictating the universe's expansion history and
        other relevant parameters.
    :type z: float, array_like, or list
    :type cosmology: astropy.cosmology instance
    :return: Growth factor at the specified redshift(s), as a float (for scalar z) or
        ndarray (for array z).
    :rtype: float or numpy.ndarray :note: The growth factor calculation is powered by
        the hmf library.
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
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Generate an array of halo redshifts based on a given comoving density.

    This function predicts halo redshifts by considering the expected number of halos at different redshifts, calculated from the differential comoving volume and halo number density. It applies a Poisson distribution to these expectations to simulate the actual distribution of halos.

    :param array_like redshift_list: List of redshifts for which to calculate halo distributions.
    :param `~astropy.units.Quantity` sky_area: Sky area under consideration, specified in square degrees.
    :param astropy.cosmology instance cosmology: Optional cosmology instance for the calculations. Defaults to a standard model if not provided.
    :param float m_min: Minimum halo mass for consideration in the calculation, in solar masses (M_sol). Optional.
    :param float m_max: Maximum halo mass for the calculation, in solar masses (M_sol). Optional.
    :param int resolution: Resolution of the mass grid for the calculations. Optional.
    :param float sigma8: Normalization parameter for the power spectrum, default is 0.81.
    :param float ns: The spectral index defining the tilt of the primordial power spectrum, default is 0.96.
    :param float omega_m: The matter density parameter of the universe, optional.
    :return: Array of halo redshifts based on the calculated distributions.
    :rtype: array
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
        sigma8=sigma8,
        ns=ns,
        omega_m=omega_m,
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
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Convert differential comoving volume element to differential number density per
    redshift interval.

    This function transitions from the concept of a comoving volume element to the
    actual number density of objects within that volume per unit redshift. It achieves
    this by taking the differential comoving volume element dV/dz and multiplying it by
    the number density of objects within a comoving volume at each redshift point.

    :param dV_dz: Differential comoving volume element as a function of redshift,
        detailing how volume changes with redshift.
    :param redshift_list: Array of redshifts corresponding to the dV_dz values, serving
        as the domain over which the function is defined.
    :param m_min: Minimum mass threshold for halos included in the number density
        calculation, in solar masses (M_sol). This parameter is optional, with a default
        setting if not provided.
    :param m_max: Maximum mass threshold for halos considered in the number density
        estimation, in solar masses (M_sol). This parameter is optional, with a default
        setting if not provided.
    :param resolution: The granularity of the grid used in the underlying mass function
        calculation, with a higher resolution offering more detailed insights. This
        parameter is optional, with a default value if omitted.
    :param cosmology: The cosmological model applied to the calculation, influencing the
        interpretation of redshifts and distances. This parameter is optional, with a
        default cosmology assumed if not specified.
    :param sigma8: Normalization parameter for the power spectrum, influencing the
        amplitude of mass density fluctuations. This parameter is optional, with a
        default value if not specified.
    :param ns: The spectral index defining the tilt of the primordial power spectrum, a
        key parameter in cosmological models. This parameter is optional, with a default
        value if not specified.
    :param omega_m: The matter density parameter of the universe, critical for
        understanding the evolution of structure. This parameter is optional and will
        default to the cosmology's value if not provided.
    :type dV_dz: ndarray
    :type redshift_list: ndarray
    :type m_min: float, optional
    :type m_max: float, optional
    :type resolution: int, optional
    :type cosmology: astropy.cosmology instance, optional
    :type sigma8: float, optional
    :type ns: float, optional
    :type omega_m: float, optional
    :return: An array representing the differential number density dn/dz, which
        quantifies the expected number of objects per redshift interval in a unit
        comoving volume.
    :rtype: ndarray :note: While the halo mass range and cosmological parameters are
        optional, specifying them enhances the accuracy and relevance of the calculated
        number density.
    """
    density = number_density_at_redshift(
        z=redshift_list,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=sigma8,
        ns=ns,
        omega_m=omega_m,
    )  # dn/dv at z; Mpc-3
    dV_dz *= density
    return dV_dz


def dndz_to_N(dN_dz, redshift_list):
    """Convert redshift distribution to total number of objects using Poisson
    statistics.

    This function transforms a differential redshift distribution dN/dz into an estimate
    of the total number of objects N. It applies Poisson statistics to model the
    inherent stochastic nature of astronomical object distributions, using the integral
    of dN/dz over the redshift range to set the mean of the Poisson distribution.

    :param dN_dz: Differential number of objects per redshift interval, serving as the
        rate of object occurrence as a function of redshift.
    :param redshift_list: Array of redshifts corresponding to the dN_dz values,
        outlining the span over which the distribution is defined.
    :type dN_dz: ndarray
    :type redshift_list: ndarray
    :return: The total number of objects expected within the given redshift
        distribution, determined by drawing from a Poisson distribution with the mean
        calculated from the integral of dN/dz.
    :rtype: int
    """
    N = np.trapz(dN_dz, redshift_list)
    N = np.random.poisson(N)
    return N


def dndz_to_redshifts(N, dN_dz, redshift_list):
    """Generate a cumulative distribution function (CDF) of redshifts using the
    trapezoidal rule.

    This function employs the cumulative trapezoidal rule to integrate a differential
    number distribution of objects dN/dz over redshift, producing a redshift cumulative
    distribution function (CDF). This CDF is then used to sample N redshifts, providing
    a statistical basis for modeling distributions of astronomical objects over
    redshift.

    :param N: The number of redshifts to sample from the cumulative distribution,
        representing the size of the resultant redshift array.
    :param dN_dz: Differential number of objects per redshift interval, indicating the
        rate at which the number of objects changes with redshift.
    :param redshift_list: Array of redshifts corresponding to the dN_dz values,
        delineating the domain over which the distribution applies.
    :type N: int
    :type dN_dz: ndarray
    :type redshift_list: ndarray
    :return: An array of N redshifts sampled according to the cumulative distribution
        derived from the differential redshift distribution dN/dz.
    :rtype: ndarray
    """
    assert len(dN_dz) == len(redshift_list)
    cdf = dN_dz  # reuse memory
    np.cumsum((dN_dz[1:] + dN_dz[:-1]) / 2 * np.diff(redshift_list), out=cdf[1:])
    cdf[0] = 0
    cdf /= cdf[-1]
    return np.interp(np.random.rand(N), cdf, redshift_list)


def v_per_redshift(redshift_list, cosmology, sky_area):
    """Calculate comoving volume per unit redshift.

    This function computes the comoving volume associated with each redshift in a given list of redshifts. The calculation considers the cosmological model specified and the area of the sky under consideration. This is crucial for understanding the volume over which astronomical surveys operate at different depths (redshifts).

    :param redshift_list: Array of redshifts for which to calculate the corresponding comoving volumes, representing the depth of an astronomical survey or observation.
    :param cosmology: The cosmological model to apply, which defines the universe's geometry and expansion history, influencing the calculation of comoving volumes. This parameter is optional, with a default cosmology used if not specified.
    :param sky_area: The area of the sky over which the volume calculations are to be applied, expressed in square degrees or as a solid angle, framing the scope of the astronomical observation or survey.
    :type redshift_list: array_like
    :type cosmology: astropy.cosmology instance, optional
    :type sky_area: `~astropy.units.Quantity`
    :return: An array detailing the comoving volume corresponding to each redshift in the input list, providing a spatial context for the distribution of astronomical objects.
    :rtype: array
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
    omega_m=None,
):
    """Sample halo masses at given redshift(s) using specified or default cosmological
    parameters and mass range.

    :param z: Redshift(s) at which to sample halo masses. Can be a single value or an
        iterable of values.
    :type z: float or iterable
    :param m_min: Minimum halo mass in solar masses. Defaults to a predetermined value
        if not specified.
    :type m_min: float, optional
    :param m_max: Maximum halo mass in solar masses. Defaults to a predetermined value
        if not specified.
    :type m_max: float, optional
    :param resolution: Resolution of the computational grid. Defaults to a predetermined
        value if not specified.
    :type resolution: int, optional
    :param cosmology: Cosmology instance to be used in calculations. Defaults to a
        predetermined cosmology if not specified.
    :type cosmology: astropy.cosmology instance, optional
    :param sigma8: Sigma8 parameter for the power spectrum normalization. Defaults to
        0.81 if not specified.
    :type sigma8: float, optional
    :param ns: Spectral index for the power spectrum. Defaults to 0.96 if not specified.
    :type ns: float, optional
    :param omega_m: Matter density parameter Omega_m. If not specified, the value from
        the cosmology instance is used.
    :type omega_m: float, optional
    :return: List of sampled halo masses for each provided redshift. Each element
        corresponds to the mass sampled at the respective redshift in the input.
    :rtype: list
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
                omega_m=omega_m,
            )
        )
    return mass


def redshift_mass_sheet_correction_array_from_comoving_density(redshift_list):
    """Create an array of redshifts for mass sheet correction calculations.

    This utility function generates a sequence of redshift values beginning at 0.025 and
    incrementing by 0.05, extending up to the maximum redshift found in the input
    redshift_list. The generated redshift array is particularly useful for computations
    related to mass sheet corrections in lensing studies, where discrete redshift values
    are needed across a specified range.

    :param redshift_list: Array of observed redshifts, the maximum of which determines
        the upper limit for the generated redshift sequence. This list provides the
        context or observational range within which the mass sheet correction
        calculations are relevant.
    :type redshift_list: ndarray
    :return: An array of redshift values starting from 0.025 and increasing in steps of
        0.05, terminating at or just below the maximum value found in redshift_list.
        This structured approach ensures a consistent set of redshifts for subsequent
        analytical steps.
    :rtype: ndarray
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
    :param z: Redshift(s) at which to calculate the first moment of kappa.
    :type z: list or np.ndarray
    :returns: A list of calculated first moments of kappa for each redshift in `z`.
    :rtype: list
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
