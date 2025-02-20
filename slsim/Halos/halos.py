from scipy import integrate
from colossus.lss import mass_function
from colossus.cosmology import cosmology as colossus_cosmo
import numpy as np
import warnings
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from slsim.Util.param_util import deg2_to_cone_angle
from slsim.Util.astro_util import (
    get_value_if_quantity,
    cone_radius_angle_to_physical_area,
)
from scipy.optimize import bisect


def colossus_halo_mass_function(m_200, cosmo, z, sigma8=0.81, ns=0.96, omega_m=None):
    """Calculate the differential halo mass function per logarithmic mass
    interval at a given redshift.

    This function leverages the Colossus library to determine the halo mass function
    based on a mass scale (m_200), a cosmological model (cosmo), and a redshift (z). The
    mass function, expressed as dn/dlnM, represents the number density of halos per log
    mass interval. Parameters sigma8 and ns can be specified to adjust the calculation.

    :param m_200: Halo mass range, in solar mass units divided by the Hubble parameter
        (M_sun/h), shown as an array. For example, can be set as m_200 =
        np.geomspace(minh, maxh, resolution_value)
    :type m_200: ndarray
    :param cosmo: astropy.cosmology instance detailing the cosmological parameters.
    :type cosmo: astropy.Cosmology instance
    :param z: Redshift for evaluating the mass function.
    :type z: float
    :param sigma8: Matter density fluctuations on a (8 h-1 Mpc), defaults to 0.81 if not
        specified.
    :type sigma8: float, optional
    :param ns: Spectral index, defaults to 0.96 if not specified.
    :type ns: float, optional
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same
        in Cosmology setting.
    :type omega_m: float, optional
    :return: The differential halo mass function dn/dlnM, in units of Mpc^-3.
    :rtype: ndarray
    :note: The `bhattacharya11` model within the Colossus framework is used for the mass function.
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
    # TODO: not stable for other cosmologies, problems like m_nu from yml pipline exist
    colossus_cosmo.setCosmology(cosmo_name="halo_cosmo", **params)
    h3 = np.power(cosmo.h, 3)
    mfunc_h3_dmpc3 = mass_function.massFunction(
        m_200, z, mdef="fof", model="bhattacharya11", q_out="dndlnM"
    )
    # in h^3*Mpc-3
    massf = mfunc_h3_dmpc3 * h3  # in Mpc-3
    return massf


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
    :type m_min: Quantity or float
    :param m_max: Maximum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :type m_max: Quantity or float
    :param resolution: Number of mass bins for the mass function calculation, converted to value if specified as an astropy Quantity.
    :type resolution: Quantity or int
    :param z: Redshift for halo mass function computation.
    :type z: float
    :param cosmology: Cosmology instance for calculating the halo mass function.
    :type cosmology: astropy.Cosmology instance
    :param sigma8: Sigma8 parameter, default is 0.81.
    :type sigma8: float, optional
    :param ns: Spectral index in Colossus cosmology settings, default is 0.96.
    :type ns: float, optional
    :param size: Number of random samples to draw. Returns a single value if None.
    :type size: int or None, optional
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same in Cosmology setting.
    :type omega_m: float

    :return: Sampled halo masses in M_sol, the number of samples is determined by the `size` parameter. Returns a single float if `size` is None.
    :rtype: ndarray or float
    :note: The Colossus library's `bhattacharya11` model is used for the mass function. Cosmology parameters from the provided instance are temporarily applied to Colossus as "halo_cosmo".
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

    CDF = integrate.cumulative_trapezoid(massf, np.log(m), initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size=size)
    return np.interp(n_uniform, CDF, m) / cosmology.h


def set_defaults_halos(
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
):
    """Set default values for parameters if not explicitly provided.

    Ensure all parameters have values, setting defaults for any that are
    unspecified. This utility function aids in initializing simulation
    or calculation setups where certain parameters might not be critical
    and can assume default values.

    :param m_min: The minimum halo mass, in solar mass units (M_sol).
        Defaults to a predetermined value if not specified.
    :type m_min: float, optional
    :param m_max: The maximum halo mass, in solar mass units (M_sol).
        Defaults to a predetermined value if not specified.
    :type m_max: float, optional
    :param resolution: Resolution of the computational grid. Assigned a
        default value if omitted.
    :type resolution: int, optional
    :param cosmology: Cosmology model instance to be used in
        calculations. If not provided, a default cosmology is used.
    :type cosmology: astropy.cosmology instance, optional
    :return: A tuple containing the parameters, with defaults applied
        where necessary.
    :rtype: tuple
    """
    # Default values
    if m_min is None:
        m_min = 1.0e12

    if m_max is None:
        m_max = 1.0e14

    if resolution is None:
        resolution = 100

    if cosmology is None:
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
    """Compute the number density of halos at various redshifts within a
    specific mass range.

    This function integrates the halo mass function over a given mass
    range to find the number density of halos per unit comoving volume
    at different redshifts. It's useful for understanding the
    distribution and evolution of halos across cosmic time.

    :param z: Redshift(s) at which to compute the number density, can be
        a single value or an array.
    :type z: float or ndarray
    :param m_min: Minimum halo mass included in the density calculation,
        in solar masses (M_sol). Optional, with a default value if not
        provided.
    :type m_min: float, optional
    :param m_max: Maximum halo mass for the density calculation, in
        solar masses (M_sol). Optional, with a default value if not
        provided.
    :type m_max: float, optional
    :param resolution: Number of mass bins for integrating the mass
        function. Optional, defaults to a predetermined value.
    :type resolution: int, optional
    :param cosmology: astropy.cosmology instance for the underlying
        cosmological model. Optional, defaults to a standard model if
        not provided.
    :type cosmology: astropy.Cosmology instance, optional
    :param sigma8: Normalization of the power spectrum, optional, with a
        default value if not specified.
    :type sigma8: float, optional
    :param ns: Spectral index for the power spectrum, optional, with a
        default value if not specified.
    :type ns: float, optional
    :param omega_m: Matter density parameter, optional, will use the
        cosmology setting if not specified.
    :type omega_m: float, optional
    :return: Number density of halos at each specified redshift, as a
        list (for scalar z) or ndarray (for array z).
    :rtype: list or ndarray :note: A warning is issued for NaN values in
        input redshifts, with a fallback to a default redshift of
        0.0001.
    """
    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults_halos(
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
        total_number_density = number_density_for_massf(massf, m_200, dndlnM=True)
        cdfs.append(total_number_density)
    return cdfs


def number_density_for_massf(massf, m, dndlnM=False):
    """Calculate the total number density of halos per cubic megaparsec (Mpc^3)
    at a certain redshift.

    This function integrates the mass function (massf) over a given mass range (m) to compute the total
    number density of halos. It can operate in two modes depending on the `dndlnM` flag: when `dndlnM` is
    True, it assumes `massf` represents the differential number density dN/dlnM; otherwise, it treats
    `massf` as the differential number density dN/dM.

    :param massf: The halo mass function, representing either dN/dM or dN/dlnM, depending on the
        value of `dndlnM`. Units are Mpc^-3 Msun^-1.
    :type massf: ndarray
    :param m: Array of halo masses (M_sun) over which to integrate the mass function. Must be the
        same length as `massf`.
    :type m: ndarray
    :param dndlnM: Flag indicating whether `massf` is given as dN/dlnM (True) or dN/dM (False).
        Optional, defaults to False.
    :type dndlnM: bool, optional

    :return: The total number density of halos per cubic megaparsec (Mpc^3) at the specified redshift.
    :rtype: float
    """
    if dndlnM:
        return integrate.trapezoid(massf, np.log(m))
    else:
        return integrate.trapezoid(massf * m, np.log(m))


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

    :param redshift_list: List of redshifts for which to calculate halo distributions.
    :type redshift_list: array_like/ np.ndarray
    :param sky_area: Sky area under consideration, specified in square degrees.
    :type sky_area: `~astropy.units.Quantity`
    :param astropy.cosmology instance cosmology: Optional cosmology instance for the calculations. Defaults to a standard model if not provided.
    :type cosmology: astropy.cosmology.Cosmology instance, optional
    :param m_min: Minimum halo mass for consideration in the calculation, in solar masses (M_sol). Optional.
    :type m_min: float, optional
    :param m_max: Maximum halo mass for the calculation, in solar masses (M_sol). Optional.
    :type m_max: float, optional
    :param resolution: Resolution of the mass grid for the calculations. Optional.
    :type resolution: int, optional
    :param sigma8: Sigma 8 for cosmology, default is 0.81.
    :type sigma8: float
    :param ns: The spectral index defining the tilt of the primordial power spectrum, default is 0.96.
    :type ns: float
    :param omega_m: The matter density parameter of the universe, optional.
    :type omega_m: float, optional

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
    """Convert differential comoving volume element to differential number
    density per redshift interval.

    This function transitions from the concept of a comoving volume element to the
    actual number density of objects within that volume per unit redshift. It achieves
    this by taking the differential comoving volume element dV/dz and multiplying it by
    the number density of objects within a comoving volume at each redshift point.

    :param dV_dz: Differential comoving volume element as a function of redshift,
        detailing how volume changes with redshift.
    :type dV_dz: ndarray
    :param redshift_list: Array of redshifts corresponding to the dV_dz values, serving
        as the domain over which the function is defined.
    :type redshift_list: ndarray
    :param m_min: Minimum mass threshold for halos included in the number density
        calculation, in solar masses (M_sol). This parameter is optional, with a default
        setting if not provided.
    :type m_min: float, optional
    :param m_max: Maximum mass threshold for halos considered in the number density
        estimation, in solar masses (M_sol). This parameter is optional, with a default
        setting if not provided.
    :type m_max: float, optional
    :param resolution: The granularity of the grid used in the underlying mass function
        calculation, with a higher resolution offering more detailed insights. This
        parameter is optional, with a default value if omitted.
    :type resolution: int, optional
    :param cosmology: The astropy.cosmology model applied to the calculation, influencing the
        interpretation of redshifts and distances. This parameter is optional, with a
        default cosmology assumed if not specified.
    :type cosmology: astropy.Cosmology instance, optional
    :param sigma8: Normalization parameter for the power spectrum, influencing the
        amplitude of mass density fluctuations. This parameter is optional, with a
        default value if not specified.
    :type sigma8: float, optional
    :param ns: The spectral index defining the tilt of the primordial power spectrum, a
        key parameter in cosmological models. This parameter is optional, with a default
        value if not specified.
    :type ns: float, optional
    :param omega_m: The matter density parameter of the universe, critical for
        understanding the evolution of structure. This parameter is optional and will
        default to the cosmology's value if not provided.
    :type omega_m: float, optional

    :return: An array representing the differential number density dn/dz, which
        quantifies the expected number of objects per redshift interval in a unit
        comoving volume.
    :rtype: ndarray

    :note: While the halo mass range and cosmological parameters are
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

    This function transforms a differential redshift distribution dN/dz
    into an estimate of the total number of objects N. It applies
    Poisson statistics to model the inherent stochastic nature of
    astronomical object distributions, using the integral of dN/dz over
    the redshift range to set the mean of the Poisson distribution.

    :param dN_dz: Differential number of objects per redshift interval,
        serving as the rate of object occurrence as a function of
        redshift.
    :type dN_dz: ndarray,
    :param redshift_list: Array of redshifts corresponding to the dN_dz
        values, outlining the span over which the distribution is
        defined.
    :type redshift_list: ndarray,
    :return: The total number of objects expected within the given
        redshift distribution, determined by drawing from a Poisson
        distribution with the mean calculated from the integral of
        dN/dz.
    :rtype: int
    """
    N = np.trapz(dN_dz, redshift_list)
    N = np.random.poisson(N)
    return N


def dndz_to_redshifts(N, dN_dz, redshift_list):
    """Generate a cumulative distribution function (CDF) of redshifts using the
    trapezoidal rule.

    This function employs the cumulative trapezoidal rule to integrate a
    differential number distribution of objects dN/dz over redshift,
    producing a redshift cumulative distribution function (CDF). This
    CDF is then used to sample N redshifts, providing a statistical
    basis for modeling distributions of astronomical objects over
    redshift.

    :param N: The number of redshifts to sample from the cumulative
        distribution, representing the size of the resultant redshift
        array.
    :type N: int
    :param dN_dz: Differential number of objects per redshift interval,
        indicating the rate at which the number of objects changes with
        redshift.
    :type dN_dz: ndarray,
    :param redshift_list: Array of redshifts corresponding to the dN_dz
        values, delineating the domain over which the distribution
        applies.
    :type redshift_list: ndarray,
    :return: An array of N redshifts sampled according to the cumulative
        distribution derived from the differential redshift distribution
        dN/dz.
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
    :type redshift_list: array_like
    :param cosmology: The astropy.cosmology model to apply, which defines the universe's geometry and expansion history, influencing the calculation of comoving volumes. This parameter is optional, with a default cosmology used if not specified.
    :type cosmology: astropy.Cosmology instance, optional
    :param sky_area: The area of the sky over which the volume calculations are to be applied, expressed in square degrees or as a solid angle, framing the scope of the astronomical observation or survey.
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
    """Sample halo masses at given redshift(s) using specified or default
    cosmological parameters and mass range.

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
    :param cosmology: astropy.cosmology instance to be used in calculations. Defaults to a
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
    ) = set_defaults_halos(
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
    :type redshift_list: ndarray:return: An array of redshift values starting from 0.025 and increasing in steps of
        0.05, terminating at or just below the maximum value found in redshift_list.
        This structured approach ensures a consistent set of redshifts for subsequent
        analytical steps.
    :rtype: ndarray
    """

    z_max = redshift_list[-1]
    linspace_values = np.arange(0.025, z_max, 0.05)
    return linspace_values


def kappa_ext_for_each_sheet(
    redshift_list, first_moment, sky_area, cosmology, z_sigma_crit_source=10.0
):
    """Calculate the external convergence (kappa_ext) for lensing sheets at
    given redshifts.

    The function computes kappa_ext using the first moment of mass, the area under consideration,
    and the critical surface mass density.

    :param redshift_list: Redshifts of the lens planes for which kappa_ext is calculated.
    :type redshift_list: list or ndarray
    :param first_moment: First moment (expected value) of mass for each redshift in the redshift_list.
    :type first_moment: list or ndarray
    :param sky_area: Area of the sky in square degrees under consideration for the lensing calculation.
    :type sky_area: `~astropy.units.Quantity`
    :param cosmology: astropy.Cosmology instance used for the calculation.
    :type cosmology: astropy.cosmology instance
    :param z_sigma_crit_source: Redshift of the source plane for the critical surface mass density calculation.
    :type z_sigma_crit_source: float, optional
    :return: Array of kappa_ext values for each redshift in the redshift_list.
    :rtype: ndarray
    """
    assert len(redshift_list) == len(first_moment)
    cone_opening_angle = deg2_to_cone_angle(sky_area.value)
    # TODO: make it possible for other geometry model
    # TODO: hard code z_souce here
    lens_cosmo = LensCosmo(
        z_lens=redshift_list, z_source=z_sigma_crit_source, cosmo=cosmology
    )
    epsilon_crit = lens_cosmo.sigma_crit

    area = cone_radius_angle_to_physical_area(
        cone_opening_angle, redshift_list, cosmology
    )  # mpc2
    first_moment_d_area = np.divide(np.array(first_moment), np.array(area))
    kappa_ext = np.divide(first_moment_d_area, epsilon_crit)
    return -kappa_ext


def expected_mass_at_redshift(
    z,
    sky_area,
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Calculate the first moment of mass at given redshift(s) using specified
    or default cosmological parameters and mass range.

    This function computes the first moment of mass within a given sky area and redshift range by
    integrating the halo mass function weighted by mass over the specified mass range.

    :param z: Redshift(s) at which to calculate the first moment of mass. Can be a single value or
        an iterable of values.
    :type z: float or iterable
    :param sky_area: Area of the sky in square degrees under consideration for the calculation.
    :type sky_area: `~astropy.units.Quantity`
    :param m_min: Minimum halo mass in solar masses for the integration range. Defaults to a
        predetermined value if not specified.
    :type m_min: float, optional
    :param m_max: Maximum halo mass in solar masses for the integration range. Defaults to a
        predetermined value if not specified.
    :type m_max: float, optional
    :param resolution: Resolution of the computational grid for the mass integration. Defaults to a
        predetermined value if not specified.
    :type resolution: int, optional
    :param cosmology: astropy.cosmology instance to be used in calculations. Defaults to a
        predetermined cosmology if not specified.
    :type cosmology: astropy.cosmology instance, optional
    :param sigma8: Sigma8. Defaults to
        0.81 if not specified.
    :type sigma8: float, optional
    :param ns: Spectral index for the power spectrum. Defaults to 0.96 if not specified.
    :type ns: float, optional
    :param omega_m: Matter density parameter Omega_m. If not specified, the value from
        the cosmology instance is used.
    :type omega_m: float, optional
    :return: List of first moments of mass for each provided redshift. Each element
        corresponds to the first moment of mass at the respective redshift in the input.
    :rtype: list
    """
    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults_halos(m_min, m_max, resolution, cosmology)
    m2_list = []
    delta_z = np.diff(z)[0]
    for zi in z:
        N = colossus_halo_expected_number_certain_bin(
            z_c=zi,
            dz=delta_z,
            sky_area=sky_area,
            m_min=m_min,
            m_max=m_max,
            resolution=resolution,
            cosmology=cosmology,
            sigma8=sigma8,
            ns=ns,
            omega_m=omega_m,
        )
        expectation_m = colossus_halo_expected_mass_sampler(
            m_min=m_min,
            m_max=m_max,
            resolution=resolution,
            z=zi,
            cosmology=cosmology,
            sigma8=sigma8,
            ns=ns,
            omega_m=omega_m,
        )
        m2 = expectation_m * N
        m2_list.append(m2)
    return m2_list


def colossus_halo_expected_mass_sampler(
    m_min,
    m_max,
    resolution,
    z,
    cosmology,
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Sample the average halo masses from a mass function within defined mass
    boundaries. (what the average mass of certain halo if it exists in redshift
    z)

    Utilizes the Colossus library to compute a halo mass function, from which halo masses are sampled. The function calculates the mass function across a specified mass range and redshift, using given cosmological parameters and, optionally, values for sigma8 and ns for power spectrum normalization and spectral index, respectively.

    :param m_min: Minimum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :type m_min: Quantity or float
    :param m_max: Maximum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :type m_max: Quantity or float
    :param resolution: Number of mass bins for the mass function calculation, converted to value if specified as an astropy Quantity.
    :type resolution: Quantity or int
    :param z: Redshift for halo mass function computation.
    :type z: float
    :param cosmology: Cosmology instance for calculating the halo mass function.
    :type cosmology: astropy.Cosmology instance
    :param sigma8: Sigma8 parameter, default is 0.81.
    :type sigma8: float, optional
    :param ns: Spectral index in Colossus cosmology settings, default is 0.96.
    :type ns: float, optional
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same in Cosmology setting.
    :type omega_m: float
    :return: Sampled halo masses in M_sol, the number of samples is determined by the `size` parameter. Returns a single float if `size` is None.
    :rtype: ndarray or float
    :note: The Colossus library's `bhattacharya11` model is used for the mass function. Cosmology parameters from the provided instance are temporarily applied to Colossus as "halo_cosmo".
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

    mass_times_massf = m * massf
    integral_mass_times_massf = integrate.cumulative_trapezoid(
        mass_times_massf, np.log(m), initial=0
    )

    integral_massf = integrate.cumulative_trapezoid(massf, np.log(m), initial=0)
    average_massh = integral_mass_times_massf[-1] / integral_massf[-1]
    return average_massh / cosmology.h


def colossus_halo_expected_number_certain_bin(
    z_c,
    dz,
    sky_area,
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Calculate the first moment of mass at given redshift(s) using specified
    or default cosmological parameters and mass range.

    This function computes the first moment of mass within a given sky area and redshift range by
    integrating the halo mass function weighted by mass over the specified mass range.
    :param z_c: center redshift for the bin
    :type z_c: float
    :param dz: redshift bin width
    :type dz: float
    :param sky_area: Area of the sky in square degrees under consideration for the calculation.
    :type sky_area: `~astropy.units.Quantity`
    :param m_min: Minimum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param m_max: Maximum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param resolution: Number of mass bins for the mass function calculation, converted to value if specified as an astropy Quantity.
    :param cosmology: Cosmology instance for calculating the halo mass function.
    :param sigma8: Sigma8 parameter, default is 0.81.
    :param ns: Spectral index in Colossus cosmology settings, default is 0.96.
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same in Cosmology setting.
    :type m_min: Quantity or float
    :type m_max: Quantity or float
    :type resolution: Quantity or int
    :type cosmology: astropy.Cosmology instance
    :type sigma8: float, optional
    :type ns: float, optional
    :type omega_m: float:return: first moment of number of halo in the bin
    :rtype: float
    """
    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults_halos(m_min, m_max, resolution, cosmology)
    redshift_list = np.linspace(z_c - dz / 2, z_c + dz / 2, 30)

    dV_dz = v_per_redshift(redshift_list, cosmology, sky_area)
    density = number_density_at_redshift(
        z=redshift_list,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=sigma8,
        ns=ns,
        omega_m=omega_m,
    )
    assert len(density) == len(redshift_list)
    dN_dz = density * dV_dz
    N = np.trapz(dN_dz, redshift_list)
    return N


def colossus_halo_expected_number(
    zmax,
    sky_area,
    m_min=None,
    m_max=None,
    resolution=None,
    cosmology=None,
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Calculate the first moment of mass at given redshift(s) using specified
    or default cosmological parameters and mass range.

    This function computes the first moment of mass within a given sky area and redshift range by
    integrating the halo mass function weighted by mass over the specified mass range.
    :param zmax: maximum-redshift
    :type zmax: float
    :param sky_area: Area of the sky in square degrees under consideration for the calculation.
    :type sky_area: `~astropy.units.Quantity`
    :param m_min: Minimum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param m_max: Maximum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param resolution: Number of mass bins for the mass function calculation, converted to value if specified as an astropy Quantity.
    :param cosmology: Cosmology instance for calculating the halo mass function.
    :param sigma8: Sigma8 parameter, default is 0.81.
    :param ns: Spectral index in Colossus cosmology settings, default is 0.96.
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same in Cosmology setting.
    :type m_min: Quantity or float
    :type m_max: Quantity or float
    :type resolution: Quantity or int
    :type cosmology: astropy.Cosmology instance
    :type sigma8: float, optional
    :type ns: float, optional
    :type omega_m: float:return: first moment of number of halo in the bin
    :rtype: float
    """
    (
        m_min,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults_halos(m_min, m_max, resolution, cosmology)
    redshift_list = np.linspace(0, zmax, 100)

    dV_dz = v_per_redshift(redshift_list, cosmology, sky_area)
    density = number_density_at_redshift(
        z=redshift_list,
        m_min=m_min,
        m_max=m_max,
        resolution=resolution,
        cosmology=cosmology,
        sigma8=sigma8,
        ns=ns,
        omega_m=omega_m,
    )
    assert len(density) == len(redshift_list)
    dN_dz = density * dV_dz
    N = np.trapz(dN_dz, redshift_list)
    return N


def optimize_min_mass_based_on_number(
    target_n_halos,
    zmax,
    sky_area,
    m_max=None,
    resolution=None,
    cosmology=None,
    sigma8=0.81,
    ns=0.96,
    omega_m=None,
):
    """Calculate the excepted minimum mass based on the target number of halos.

    :param zmax: maximum-redshift
    :type zmax: float
    :param target_n_halos: target number of halos
    :type target_n_halos: float
    :param sky_area: Area of the sky in square degrees under consideration for the calculation.
    :type sky_area: `~astropy.units.Quantity`
    :param m_max: Maximum halo mass, in M_sol/h. Converted to value if specified as an astropy Quantity.
    :param resolution: Number of mass bins for the mass function calculation, converted to value if specified as an astropy Quantity.
    :param cosmology: Cosmology instance for calculating the halo mass function.
    :param sigma8: Sigma8 parameter, default is 0.81.
    :param ns: Spectral index in Colossus cosmology settings, default is 0.96.
    :param omega_m: Omega_m in Cosmology, defaults to none which will lead to the same in Cosmology setting.
    :type resolution: Quantity or int
    :type cosmology: astropy.Cosmology instance
    :type sigma8: float, optional
    :type ns: float, optional
    :type omega_m: float:return: first moment of number of halo in the bin
    :rtype: float
    """
    (
        _,
        m_max,
        resolution,
        cosmology,
    ) = set_defaults_halos(1e10, m_max, resolution, cosmology)

    def _difference(m_min):
        n_halos = colossus_halo_expected_number(
            zmax=zmax,
            sky_area=sky_area,
            m_min=m_min,
            m_max=m_max,
            resolution=resolution,
            cosmology=cosmology,
            sigma8=sigma8,
            ns=ns,
            omega_m=omega_m,
        )
        return n_halos - target_n_halos

    try:
        result_m_min = bisect(_difference, 1e9, 1e14, xtol=1e6)
        return result_m_min
    except Exception:
        return 1e9
