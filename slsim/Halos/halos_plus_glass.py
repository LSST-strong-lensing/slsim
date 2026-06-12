import numpy as np
from slsim.Pipelines.halos_pipeline import HalosSkyPyPipeline
from slsim.Halos.halos_lens_base import HalosLensBase
from slsim.Halos.halos_statistics import HalosStatistics
from astropy.cosmology import FlatLambdaCDM
import time
from scipy import stats
import warnings
from multiprocessing import get_context
from slsim.Halos.halos_util import convergence_mean_0


def read_glass_data(file_name="kgdata.npy"):
    """Loads a numpy data file and extracts the kappa and gamma values.

    :param file_name: The name of the file to load. Defaults to
        "kgdata.npy".
    :type file_name: str, optional
    :returns: A tuple containing two numpy arrays for kappa and gamma
        values, and nside of the data.
    :rtype: tuple
    """

    def read_data_file(file_name):
        try:
            data = np.load(file_name)
            return data
        except FileNotFoundError:
            raise ValueError(
                f"Error: The file {file_name} could not be read. Please check the file path and try again."
            )

    data_array = read_data_file(file_name)
    kappa_values = data_array[:, 0]
    gamma_values = data_array[:, 1]
    nside = int(np.sqrt(len(kappa_values) / 12))
    return kappa_values, gamma_values, nside


def generate_samples_from_glass(kappa_values, gamma_values, n=10000):
    """This function fits a Gaussian Kernel Density Estimation (KDE) to the
    joint distribution of kappa and gamma values, and then generates a random
    sample from this distribution.

    :param kappa_values: The kappa values.
    :type kappa_values: numpy.ndarray
    :param gamma_values: The gamma values.
    :type gamma_values: numpy.ndarray
    :param n: The number of random numbers to generate. Defaults to
        10000.
    :type n: int, optional
    :returns: A tuple containing two numpy arrays for the randomly
        resampled kappa and gamma values.
    :rtype: tuple
    """

    kernel = stats.gaussian_kde(np.vstack([kappa_values, gamma_values]))
    kappa_random_glass, gamma_random_glass = kernel.resample(n)
    return kappa_random_glass, gamma_random_glass
    # TODO: make n more reasonable (maybe write some function relate n with bandwidth of kde)


def skyarea_form_n(nside, deg2=True):
    """This function calculates the sky area corresponding to each pixel when
    the sky is divided into 12 * nside ** 2 equal areas.
    4*pi*(180/pi)^2=129600/pi deg^2; 4*pi*(180/pi)^2*(3600)^2=1679616000000/pi
    arcsec^2.

    :param nside: The HEALPix resolution parameter. The total number of
        pixels is 12 * nside ** 2.
    :type nside: int
    :param deg2: If True, the sky area is returned in square degrees. If
        False, the sky area is returned in square arcseconds. Defaults
        to True.
    :type deg2: bool, optional
    :return: The sky area per pixel in either degree^2 or arcsecond^2,
        depending on the value of deg2.
    :rtype: float
    """

    if deg2:
        skyarea = 129600 / (12 * np.pi * nside**2)  # sky area in square degrees
    else:
        skyarea = 1679616000000 / (
            12 * np.pi * nside**2
        )  # sky area in square arcseconds
    return skyarea


def generate_maps_kmean_zero_using_halos(
    skypy_config=None,
    skyarea=0.0001,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=None,
    samples_number_for_one_halos=1000,
    renders_numbers=500,
):
    """Given the specified parameters, this function repeatedly renders the
    same halo list (same z & m) for different positions
    (`samples_number_for_one_halos`) times and then ensures that the mean of
    the kappa values is zero. It then gets the kde of the weak- lensing
    distribution for this halo list and resamples `renders_numbers` sets of the
    corresponding convergence (`kappa`) and shear (`gamma`) values for this
    halo list.

    :param skypy_config: Path to SkyPy configuration yaml file. If None,
        the default skypy configuration file is used.
    :type skypy_config: str or None
    :param skyarea: The sky area in square degrees. Default is 0.0001.
    :type skyarea: float, optional
    :param cosmo: The cosmological model. If None, a default
        FlatLambdaCDM model with H0=70 and Om0=0.3 is used.
    :type cosmo: astropy.cosmology.FlatLambdaCDM, optional
    :param m_min: Minimum halo mass for the pipeline. Default is None.
    :type m_min: float or None, optional
    :param m_max: Maximum halo mass for the pipeline. Default is None.
    :type m_max: float or None, optional
    :param z_max: Maximum redshift for the pipeline. Default is None.
    :type z_max: float or None, optional
    :param samples_number_for_one_halos: The number of samples for each
        halo. Default is 1000.
    :type samples_number_for_one_halos: int, optional
    :param renders_numbers: The number of random numbers to generate.
        Default is 500.
    :type renders_numbers: int, optional
    :returns: A tuple containing the randomly resampled kappa values and
        gamma values.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM

        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    sky_area = skyarea

    pipeline = HalosSkyPyPipeline(
        sky_area=sky_area,
        skypy_config=skypy_config,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
    )
    halos = pipeline.halos
    mass_sheet_correction = pipeline.mass_sheet_correction

    halos_lens = HalosStatistics(
        halos_list=halos,
        mass_correction_list=mass_sheet_correction,
        sky_area=sky_area,
        cosmo=cosmo,
        samples_number=samples_number_for_one_halos,
    )
    kappa_gamma_distribution = halos_lens.get_kappa_gamma_distib(gamma_tot=True)

    kappa_gamma_distribution = np.array(kappa_gamma_distribution)
    kappa_values_halos = kappa_gamma_distribution[:, 0]
    gamma_values_halos = kappa_gamma_distribution[:, 1]
    modified_kappa_halos = convergence_mean_0(kappa_values_halos)

    # Check if the values are all zeros
    if np.all(modified_kappa_halos == 0) and np.all(gamma_values_halos == 0):
        # Return arrays of zeros with the same shape
        return np.array([0] * renders_numbers), np.array([0] * renders_numbers)

    kernel = stats.gaussian_kde(np.vstack([modified_kappa_halos, gamma_values_halos]))
    kappa_random_halos, gamma_random_halos = kernel.resample(renders_numbers)
    return kappa_random_halos, gamma_random_halos

    # TODO: make samples_number_for_one_halos & renders_numbers more reasonable (maybe write some function relate them
    #  with bandwidth of kde)


def generate_meanzero_halos_multiple_times(
    n_times=20,
    skypy_config=None,
    skyarea=0.0001,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=None,
    samples_number_for_one_halos=1000,
    renders_numbers=500,
):
    """Given the specified parameters, this function repeatedly renders the
    same halo list (same z & m) for different positions
    (`samples_number_for_one_halos`) times and then ensures that the mean of
    the kappa values is zero. It then gets the kde of the weak- lensing
    distribution for this halo list and resamples `renders_numbers` sets of the
    corresponding convergence (`kappa`) and shear (`gamma`) values for this
    halo list. This process is repeated `n_times` to accumulate the results.

    :param n_times: The number of times to repeat the generation of
        kappa and gamma values. Default is 20.
    :type n_times: int, optional
    :param skypy_config: Path to SkyPy configuration yaml file. If None,
        the default skypy configuration file is used.
    :type skypy_config: string or None
    :param skyarea: The sky area in square degrees. Default is 0.0001.
    :type skyarea: float, optional
    :param cosmo: The cosmological model. If None, a default
        FlatLambdaCDM model with H0=70 and Om0=0.3 is used.
    :type cosmo: astropy.cosmology.FlatLambdaCDM, optional
    :param m_min: Minimum halo mass for the pipeline. Default is None.
    :type m_min: float or None, optional
    :param m_max: Maximum halo mass for the pipeline. Default is None.
    :type m_max: float or None, optional
    :param z_max: Maximum redshift for the pipeline. Default is None.
    :type z_max: float or None, optional
    :param samples_number_for_one_halos: The number of samples for each
        halo. Default is 1000.
    :type samples_number_for_one_halos: int, optional
    :param renders_numbers: The number of random numbers to generate.
        Default is 500.
    :type renders_numbers: int, optional
    :returns: A tuple containing the accumulated randomly resampled
        kappa values and gamma values.
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    accumulated_kappa_random_halos = []
    accumulated_gamma_random_halos = []
    n_times_range = range(n_times)
    start_time = time.time()
    for _ in n_times_range:
        kappa_random_halos, gamma_random_halos = generate_maps_kmean_zero_using_halos(
            skypy_config=skypy_config,
            skyarea=skyarea,
            cosmo=cosmo,
            samples_number_for_one_halos=(samples_number_for_one_halos),
            renders_numbers=renders_numbers,
            m_min=m_min,
            m_max=m_max,
            z_max=z_max,
        )
        accumulated_kappa_random_halos.extend(kappa_random_halos)
        accumulated_gamma_random_halos.extend(gamma_random_halos)

    end_time = time.time()  # Note the end time
    print(f"Elapsed time: {end_time - start_time} seconds")

    return np.array(accumulated_kappa_random_halos), np.array(
        accumulated_gamma_random_halos
    )


def halos_plus_glass(
    kappa_random_halos, gamma_random_halos, kappa_random_glass, gamma_random_glass
):
    r"""Combine the kappa and gamma values from Halos and Glass distributions,
    returning the combined values. For Halos kappa, it is suggested to use
    modified kappa values (i.e., direct kappa minus mean(kappa)).

    :param kappa_random_halos: The randomly resampled kappa values from the Halos.
    :type kappa_random_halos: numpy.ndarray
    :param gamma_random_halos: The randomly resampled gamma values from the Halos.
    :type gamma_random_halos: numpy.ndarray
    :param kappa_random_glass: The randomly resampled kappa values from Glass.
    :type kappa_random_glass: numpy.ndarray
    :param gamma_random_glass: The randomly resampled gamma values from Glass.
    :type gamma_random_glass: numpy.ndarray
    :return: A tuple containing two numpy arrays for the combined kappa and gamma values.
    :rtype: tuple

    .. note::
        The total shear, :math:`\gamma_{\text{tot}}`, is computed based on the relationships:

        .. math:: \gamma_{\text{tot}} = \sqrt{(\gamma1_{\text{Halos}}  + \gamma1_{\text{GLASS}})^2 + (\gamma2_{\text{
            Halos}}  + \gamma2_{\text{GLASS}})^2}

        and

        .. math:: \gamma_{\text{tot}} = \sqrt{\gamma_{\text{Halos}}^2 + \gamma_{\text{GLASS}}^2 + 2\gamma_{\text{
            Halos}}\gamma_{\text{GLASS}} \times \cos(\alpha-\beta)}

        Given that :math:`\alpha` and :math:`\beta` are randomly distributed, and their difference, :math:`\alpha-\beta`,
        follows a normal distribution, the shear is given by:

        .. math:: \gamma_{\text{tot}}^2 = \gamma_{\text{halos}}^2 + \gamma_{\text{GLASS}}^2 + 2\gamma_{\text{
            halos}}\gamma_{\text{GLASS}}\cdot \cos(\text{random angle})
    """

    total_kappa = [
        k_h + k_g for k_h, k_g in zip(kappa_random_halos, kappa_random_glass)
    ]
    random_angles = np.random.uniform(0, 2 * np.pi, size=len(gamma_random_halos))
    random_numbers = np.cos(random_angles)
    total_gamma = [
        np.sqrt((g_h**2) + (g_g**2) + (2 * r * g_h * g_g))
        for g_h, g_g, r in zip(gamma_random_halos, gamma_random_glass, random_numbers)
    ]

    return total_kappa, total_gamma
    # TODO: Add math


def run_halos_without_kde(
    n_iterations=1,
    sky_area=0.0001,
    samples_number=1500,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=5.0,
    mass_sheet_correction=True,
    listmean=False,
):
    """Under the specified `sky_area`, generate `n_iterations` sets of halo
    lists. For each set, simulate `samples_number` times to obtain the
    convergence (`kappa`) and shear (`gamma`) values at the origin. These
    values are directly appended without any additional processing (i.e.,
    without KDE computation, resampling, or subtracting the mean kappa value).

    :param n_iterations: Number of iterations or halo lists to generate.
    :type n_iterations: int, optional
    :param sky_area: Total sky area (in steradians) over which Halos are distributed.
    :type sky_area: float, optional
    :param samples_number: Number of samples for statistical calculations.
    :type samples_number: int, optional
    :param cosmo: Cosmology used for the simulations. If not provided, the default astropy cosmology is used.
    :type cosmo: astropy.cosmology instance, optional
    :param m_min: Minimum mass of the Halos to consider. If not provided, no lower limit is set.
    :type m_min: float, optional
    :param m_max: Maximum mass of the Halos to consider. If not provided, no upper limit is set.
    :type m_max: float, optional
    :param z_max: Maximum redshift of the Halos to consider. If not provided, no upper limit is set.
    :type z_max: float, optional
    :param mass_sheet_correction: Flag to apply mass sheet correction. Defaults to True.
    :type mass_sheet_correction: bool, optional
    :param listmean: Flag to subtract the mean kappa value from the results. Defaults to False.
    :type listmean: bool, optional
    :returns: A tuple containing two lists; the first list contains the combined convergence (`kappa`) values from all iterations, and the second list contains the combined shear (`gamma`) values from all iterations.
    :rtype: (list, list)

    .. note::
        This function initializes a halo distribution pipeline for each iteration, simulates Halos,
        and then computes the lensing properties (`kappa` and `gamma`). The results from all iterations
        are concatenated to form the final output lists.

    .. warning::
        If no cosmology is provided, the function uses the default astropy cosmology, which is a flat
        Lambda-CDM model with H0=70 and Om0=0.3.
    """

    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM

        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kappa_values_total = []
    gamma_values_total = []
    # Show progress only when n_iterations > 30
    iter_range = range(n_iterations)

    start_time = time.time()  # Note the start time
    for _ in iter_range:
        npipeline = HalosSkyPyPipeline(
            sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
        )
        nhalos = npipeline.halos
        if mass_sheet_correction:
            mass_sheet_correction_list = npipeline.mass_sheet_correction
            nhalos_lens = HalosStatistics(
                halos_list=nhalos,
                mass_correction_list=mass_sheet_correction_list,
                sky_area=sky_area,
                cosmo=cosmo,
                samples_number=samples_number,
                mass_sheet=True,
                z_source=z_max,
            )

            nkappa_gamma_distribution = nhalos_lens.get_kappa_gamma_distib(
                gamma_tot=True, listmean=listmean
            )
            nkappa_gamma_distribution = np.array(
                nkappa_gamma_distribution
            )  # Convert list of lists to numpy array

            nkappa_values_halos = nkappa_gamma_distribution[:, 0]
            ngamma_values_halos = nkappa_gamma_distribution[:, 1]

            kappa_values_total.extend(nkappa_values_halos)
            gamma_values_total.extend(ngamma_values_halos)
        else:
            nhalos_lens = HalosStatistics(
                halos_list=nhalos,
                sky_area=sky_area,
                cosmo=cosmo,
                samples_number=samples_number,
                mass_sheet=False,
                z_source=z_max,
            )

            nkappa_gamma_distribution = nhalos_lens.get_kappa_gamma_distib(
                gamma_tot=True, listmean=listmean
            )
            nkappa_gamma_distribution = np.array(
                nkappa_gamma_distribution
            )  # Convert list of lists to numpy array

            nkappa_values_halos = nkappa_gamma_distribution[:, 0]
            ngamma_values_halos = nkappa_gamma_distribution[:, 1]

            kappa_values_total.extend(nkappa_values_halos)
            gamma_values_total.extend(ngamma_values_halos)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )

    return kappa_values_total, gamma_values_total


def worker_run_halos_without_kde(
    iter_num,
    sky_area,
    m_min,
    m_max,
    z_max,
    cosmo,
    samples_number,
    mass_sheet_correction,
    listmean,
    sigma_8,
    omega_m,
):
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
        sigma_8=sigma_8,
        omega_m=omega_m,
    )
    nhalos = npipeline.halos
    if mass_sheet_correction:
        mass_sheet_correction_list = npipeline.mass_sheet_correction
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            mass_correction_list=mass_sheet_correction_list,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max,
        )
    nkappa_gamma_distribution = (
        nhalos_lens.get_kappa_gamma_distib_without_multiprocessing(
            gamma_tot=True, listmean=listmean
        )
    )
    nkappa_gamma_distribution = np.array(nkappa_gamma_distribution)

    nkappa_values_halos = nkappa_gamma_distribution[:, 0]
    ngamma_values_halos = nkappa_gamma_distribution[:, 1]

    return nkappa_values_halos, ngamma_values_halos


def run_halos_without_kde_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    samples_number=1500,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=5.0,
    mass_sheet_correction=True,
    listmean=False,
    sigma_8=0.81,
    omega_m=None,
):
    """Under the specified `sky_area`, generate `n_iterations` sets of halo
    lists. For each set, simulate `samples_number` times to obtain the
    convergence (`kappa`) and shear (`gamma`) values at the origin. These
    values are directly appended without any additional processing (i.e.,
    without KDE computation, resampling, or subtracting the mean kappa value),
    the process procedure `n_iterations` generate with multiprocessing.

    :param n_iterations: Number of iterations or halo lists to generate. Defaults to 1.
    :type n_iterations: int, optional
    :param sky_area: Total sky area (in steradians) over which Halos are distributed. Defaults to 0.0001 steradians.
    :type sky_area: float, optional
    :param samples_number: Number of samples for statistical calculations. Defaults to 1500.
    :type samples_number: int, optional
    :param cosmo: Cosmology used for the simulations. If not provided, the default astropy cosmology is used.
    :type cosmo: astropy.cosmology instance, optional
    :param m_min: Minimum mass of the Halos to consider. If not provided, no lower limit is set.
    :type m_min: float, optional
    :param m_max: Maximum mass of the Halos to consider. If not provided, no upper limit is set.
    :type m_max: float, optional
    :param z_max: Maximum redshift of the Halos to consider. If not provided, no upper limit is set.
    :type z_max: float, optional
    :param mass_sheet_correction: If True, apply mass sheet correction. Defaults to True.
    :type mass_sheet_correction: bool, optional
    :param listmean: If True, subtract the mean kappa value from the results. Defaults to False.
    :type listmean: bool, optional
    :param sigma_8: The value of sigma_8 for the cosmology. Defaults to 0.81.
    :type sigma_8: float, optional
    :param omega_m: The value of omega_m for the cosmology. If None, the default value is used.
    :type omega_m: float, optional
    :returns: A tuple containing two lists; the first list contains the combined convergence (`kappa`) values from all iterations, and the second list contains the combined shear (`gamma`) values from all iterations.
    :rtype: (list, list)

    .. note::
        This function initializes a halo distribution pipeline for each iteration, simulates Halos,
        and then computes the lensing properties (`kappa` and `gamma`). The results from all iterations
        are concatenated to form the final output lists.

    .. warning::
        If no cosmology is provided, the function uses the default astropy cosmology, which is a flat
        Lambda-CDM model with H0=70 and Om0=0.3.
    """

    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kappa_values_total = []
    gamma_values_total = []

    start_time = time.time()  # Note the start time

    args = [
        (
            i,
            sky_area,
            m_min,
            m_max,
            z_max,
            cosmo,
            samples_number,
            mass_sheet_correction,
            listmean,
            sigma_8,
            omega_m,
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_run_halos_without_kde, args)

    for nkappa, ngamma in results:
        kappa_values_total.extend(nkappa)
        gamma_values_total.extend(ngamma)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return kappa_values_total, gamma_values_total


def run_kappaext_gammaext_kde_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    samples_number=1,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=5.0,
    mass_sheet_correction=True,
    listmean=False,
    output_format="dict",
):
    """Run the kappa-gamma external convergence distribution for a given number
    of iterations using multiprocessing.

    This function generates kappa and gamma distributions using the provided parameters and
    a worker function (`worker_kappaext_gammaext_kde`). The distributions are generated
    in parallel over the specified number of iterations.

    :param n_iterations: The number of iterations to run the simulation. Defaults to 1.
    :type n_iterations: int, optional
    :param sky_area: The area of the sky under consideration (in steradians). Defaults to 0.0001.
    :type sky_area: float, optional
    :param samples_number: The number of samples to be used. Defaults to 1.
    :type samples_number: int, optional
    :param cosmo: The cosmology to be used. If not provided, it defaults to FlatLambdaCDM with H0=70 and Om0=0.3.
    :type cosmo: astropy.cosmology instance, optional
    :param m_min: The minimum mass for the simulation. If None, it is decided by the worker function.
    :type m_min: float, optional
    :param m_max: The maximum mass for the simulation. If None, it is decided by the worker function.
    :type m_max: float, optional
    :param z_max: The maximum redshift for the simulation. If None, it is decided by the worker function.
    :type z_max: float, optional
    :param mass_sheet_correction: If True, apply mass sheet correction. Defaults to True.
    :type mass_sheet_correction: bool, optional
    :param output_format: The format in which the results should be returned, either as `dict` or `vector`. Defaults to `dict`.
    :type output_format: str, optional
    :return: A list of kappa and gamma values across all iterations.
    :rtype: list

    .. note::
        The function employs multiprocessing to run simulations in parallel, improving computational efficiency.
        The elapsed runtime for the simulations is printed to the console.
    """

    # TODO: BUG
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    kappaext_gammaext_values_total = []

    start_time = time.time()  # Note the start time

    args = [
        (
            i,
            sky_area,
            m_min,
            m_max,
            z_max,
            cosmo,
            samples_number,
            mass_sheet_correction,
            listmean,
            output_format,
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_kappaext_gammaext_kde, args)

    for generate_distributions_0to5 in results:
        kappaext_gammaext_values_total.extend(generate_distributions_0to5)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return kappaext_gammaext_values_total


def worker_kappaext_gammaext_kde(
    iter_num,
    sky_area,
    m_min,
    m_max,
    z_max,
    cosmo,
    samples_number,
    mass_sheet_correction,
    listmean,
    output_format,
):
    """Worker function that generates kappa-gamma distributions for given
    parameters.

    This function utilizes the `HalosSkyPyPipeline` to generate halos and, if necessary, mass sheet
    corrections. It then uses these halos (and corrections) to construct a `HalosLens` object and
    generate kappa-gamma distributions.

    :param iter_num: The iteration number, mainly used for tracking in a parallel processing context.
    :type iter_num: int
    :param sky_area: The area of the sky under consideration (in steradians).
    :type sky_area: float
    :param m_min: The minimum mass for the simulation.
    :type m_min: float
    :param m_max: The maximum mass for the simulation.
    :type m_max: float
    :param z_max: The maximum redshift for the simulation.
    :type z_max: float
    :param cosmo: The cosmology to be used.
    :type cosmo: astropy.cosmology.FlatLambdaCDM
    :param samples_number: The number of samples to be used.
    :type samples_number: int
    :param mass_sheet_correction: If True, apply mass sheet correction.
    :type mass_sheet_correction: bool
    :param output_format: The format in which the results should be returned, either as `dict` or `vector`. Defaults to `dict`.
    :type output_format: str, optional
    :param listmean: If True, subtract the mean kappa value from the results.
    :type listmean: bool
    :returns: A list of kappa and gamma values for the specified parameters.
    :rtype: list

    This function is primarily intended to be used as a worker function in a parallel processing
    framework, where multiple instances of the function can be run simultaneously.
    """

    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max,
        )

    distributions_0to5 = nhalos_lens.generate_distributions_0to5(
        output_format=output_format, listmean=listmean
    )
    # TODO: BUG
    return distributions_0to5


def run_certain_redshift_lensext_kde_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    samples_number=1,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=None,
    mass_sheet_correction=True,
    zs=None,
    zd=None,
    listmean=False,
    sigma_8=0.81,
    omega_m=None,
):
    """Generates distributions of kappa_ext and gamma_ext for a given redshift
    and cosmology.

    :param n_iterations: Number of iterations for multiprocessing to run
        the simulation.
    :type n_iterations: int
    :param sky_area: Sky area over which halos are sampled.
    :type sky_area: float
    :param m_min: Minimum halo mass.
    :type m_min: float
    :param m_max: Maximum halo mass.
    :type m_max: float
    :param z_max: Maximum redshift value.
    :type z_max: float
    :param cosmo: Cosmology used for the calculations.
    :param samples_number: Number of samples to generate.
    :type samples_number: int
    :param mass_sheet_correction: Flag to apply mass sheet correction.
    :type mass_sheet_correction: bool
    :param zs: Source redshift.
    :type zs: float
    :param zd: Lens redshift.
    :type zd: float
    :param listmean: Flag to return the mean of the list.
    :type listmean: bool
    :param sigma_8: Sigma_8 parameter for the cosmology.
    :type sigma_8: float
    :param omega_m: Omega matter parameter for the cosmology.
    :type omega_m: float
    :return: Distributions of kappa_ext and gamma_ext.
    :rtype: np.ndarray
    """
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if zs is None:
        zs = 1.5
    if zd is None:
        zd = 1.0
    if z_max is None:
        z_max = 5.0

    kappaext_gammaext_values = []

    start_time = time.time()  # Note the start time

    args = [
        (
            i,
            sky_area,
            m_min,
            m_max,
            z_max,
            cosmo,
            samples_number,
            mass_sheet_correction,
            zs,
            zd,
            listmean,
            sigma_8,
            omega_m,
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_certain_redshift_lensext_kde, args)

    for distributions in results:
        kappaext_gammaext_values.extend(distributions)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return kappaext_gammaext_values


def worker_certain_redshift_lensext_kde(
    iter_num,
    sky_area,
    m_min,
    m_max,
    z_max,
    cosmo,
    samples_number,
    mass_sheet_correction,
    zs,
    zd,
    listmean,
    sigma_8,
    omega_m,
):
    """Generates distributions of kappa_ext and gamma_ext for a given redshift
    and cosmology.

    :param iter_num: Iteration number, used for multiprocessing.
    :type iter_num: int
    :param sky_area: Sky area over which halos are sampled.
    :type sky_area: float
    :param m_min: Minimum halo mass.
    :type m_min: float
    :param m_max: Maximum halo mass.
    :type m_max: float
    :param z_max: Maximum redshift value.
    :type z_max: float
    :param cosmo: Cosmology used for the calculations.
    :param samples_number: Number of samples to generate.
    :type samples_number: int
    :param mass_sheet_correction: Flag to apply mass sheet correction.
    :type mass_sheet_correction: bool
    :param zs: Source redshift.
    :type zs: float
    :param zd: Lens redshift.
    :type zd: float
    :param listmean: Flag to return the mean of the list.
    :type listmean: bool
    :param sigma_8: Sigma_8 parameter for the cosmology.
    :type sigma_8: float
    :param omega_m: Omega matter parameter for the cosmology.
    :type omega_m: float
    :return: Distributions of kappa_ext and gamma_ext.
    :rtype: np.ndarray
    """

    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
        sigma_8=sigma_8,
        omega_m=omega_m,
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
            mass_sheet=True,
        )
    else:
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max,
        )
    distributions = nhalos_lens.get_kappaext_gammaext_distib_zdzs(
        zd, zs, listmean=listmean
    )
    return distributions


def run_certain_redshift_many_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    samples_number=1,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=None,
    mass_sheet_correction=True,
    zs=None,
    zd=None,
):
    """Runs the simulation for a certain redshift range multiple times using
    multiprocessing.

    :param n_iterations: Number of iterations to run the simulation,
        defaults to 1
    :type n_iterations: int, optional
    :param sky_area: The sky area over which to simulate, defaults to
        0.0001
    :type sky_area: float, optional
    :param samples_number: Number of samples to generate, defaults to 1
    :type samples_number: int, optional
    :param cosmo: Cosmology used for the simulation, defaults to None.
        If None, uses FlatLambdaCDM with H0=70, Om0=0.3.
    :type cosmo: astropy.cosmology.FLRW, optional
    :param m_min: Minimum mass for the halos, defaults to None
    :type m_min: float, optional
    :param m_max: Maximum mass for the halos, defaults to None
    :type m_max: float, optional
    :param z_max: Maximum redshift for the halos, defaults to None. If
        None, uses 5.0.
    :type z_max: float, optional
    :param mass_sheet_correction: Flag to apply mass sheet correction,
        defaults to True
    :type mass_sheet_correction: bool, optional
    :param zs: Source redshift, defaults to None. If None, uses 1.5.
    :type zs: float, optional
    :param zd: Lens redshift, defaults to None. If None, uses 1.0.
    :type zd: float, optional
    :return: A tuple containing two lists: kappaext_gammaext_values and
        lensinstance_values.
    :rtype: tuple(list, list)
    """

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if zs is None:
        zs = 1.5
    if zd is None:
        zd = 1.0
    if z_max is None:
        z_max = 5.0

    kappaext_gammaext_values = []
    lensinstance_values = []

    start_time = time.time()  # Note the start time

    args = [
        (
            i,
            sky_area,
            m_min,
            m_max,
            z_max,
            cosmo,
            samples_number,
            mass_sheet_correction,
            zs,
            zd,
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_certain_redshift_many, args)

    for distributions, lensinstance in results:
        kappaext_gammaext_values.extend(distributions)
        lensinstance_values.extend(lensinstance)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return kappaext_gammaext_values, lensinstance_values


def worker_certain_redshift_many(
    iter_num,
    sky_area,
    m_min,
    m_max,
    z_max,
    cosmo,
    samples_number,
    mass_sheet_correction,
    zs,
    zd,
):
    """
    :param iter_num: The iteration number.
    :type iter_num: int
    :param sky_area: The area of the sky under consideration.
    :type sky_area: float
    :param m_min: The minimum mass of halos to consider.
    :type m_min: float
    :param m_max: The maximum mass of halos to consider.
    :type m_max: float
    :param z_max: The maximum redshift to consider.
    :type z_max: float
    :param cosmo: The cosmological model to use.
    :type cosmo: FlatLambdaCDM
    :param samples_number: The number of samples to generate.
    :type samples_number: int
    :param mass_sheet_correction: Flag to apply mass sheet correction.
    :type mass_sheet_correction: bool
    :param zs: The source redshift.
    :type zs: float
    :param zd: The lens redshift.
    :type zd: float
    :returns: A tuple containing the distributions of parameters and the lens instance.
    :rtype: tuple
    """

    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosStatistics(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max,
        )
    distributions, lensinstance = nhalos_lens.get_all_pars_distib(zd, zs)
    return distributions, lensinstance


def run_total_mass_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=None,
):
    """Calculate the total mass of halos over multiple iterations using
    multiprocessing.

    :param n_iterations: Number of iterations to run, defaults to 1
    :type n_iterations: int, optional
    :param sky_area: The sky area over which to calculate the halo mass, defaults to
        0.0001
    :type sky_area: float, optional
    :param cosmo: The cosmology model to use, defaults to None. If None, uses
        FlatLambdaCDM with H0=70, Om0=0.3
    :type cosmo: astropy.cosmology.FlatLambdaCDM, optional
    :param m_min: The minimum mass of halos to consider, defaults to None
    :type m_min: float, optional
    :param m_max: The maximum mass of halos to consider, defaults to None
    :type m_max: float, optional
    :param z_max: The maximum redshift to consider, defaults to None. If None, uses 5.0
    :type z_max: float, optional
    :return: A list of total mass values for each iteration.
    :rtype: list
    """

    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if z_max is None:
        z_max = 5.0
        warnings.warn("No maximum redshift provided, instead uses 5.0")

    total_mass = []

    start_time = time.time()  # Note the start time

    args = [
        (
            i,
            sky_area,
            m_min,
            m_max,
            z_max,
            cosmo,
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_run_total_mass_by_multiprocessing, args)
        total_mass.extend(results)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return total_mass


def worker_run_total_mass_by_multiprocessing(
    iter_num,
    sky_area,
    m_min,
    m_max,
    z_max,
    cosmo,
):
    """Calculate the total mass of halos for a given iteration, sky area, mass
    range, maximum redshift, and cosmology.

    :param iter_num: Iteration number.
    :type iter_num: int
    :param sky_area: Sky area over which Halos are sampled. Must be in units of solid angle.
    :type sky_area: `~astropy.units.Quantity`
    :param m_min: Minimum halo mass.
    :type m_min: float
    :param m_max: Maximum halo mass.
    :type m_max: float
    :param z_max: Maximum redshift value in z_range.
    :type z_max: float
    :param cosmo: Cosmology model to use.
    :type cosmo: astropy.cosmology.FlatLambdaCDM
    :return: Total mass of halos.
    :rtype: float
    """

    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    nhalos_lens = HalosStatistics(
        halos_list=nhalos,
        sky_area=sky_area,
        cosmo=cosmo,
        samples_number=1,
        z_source=z_max,
        mass_sheet=False,
    )

    total_mass = nhalos_lens.total_halo_mass()
    return total_mass


def run_total_kappa_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    diff=0.0000001,
    num_points=500,
    diff_method="square",
    cosmo=None,
    m_min=None,
    m_max=None,
    z_max=None,
):
    """Runs the total kappa computation in parallel using multiprocessing for a
    given number of iterations.

    :param n_iterations: The number of iterations to run the computation
        for.
    :type n_iterations: int, optional
    :param sky_area: The sky area over which to compute the total kappa.
    :type sky_area: float, optional
    :param diff: The differential to use in the computation.
    :type diff: float, optional
    :param num_points: The number of points to use in the computation.
    :type num_points: int, optional
    :param diff_method: The method to use for the differential
        computation.
    :type diff_method: str, optional
    :param cosmo: The cosmology model to use. If None, a default
        FlatLambdaCDM model is used.
    :type cosmo: astropy.cosmology.FLRW, optional
    :param m_min: The minimum mass to consider in the halo computation.
    :type m_min: float, optional
    :param m_max: The maximum mass to consider in the halo computation.
    :type m_max: float, optional
    :param z_max: The maximum redshift to consider. If None, a default
        value of 5.0 is used.
    :type z_max: float, optional
    :return: A list of average kappa values for each iteration.
    :rtype: list
    """

    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if z_max is None:
        z_max = 5.0

    average_kappa_list = []

    start_time = time.time()  # Note the start time

    args = [
        (
            i,
            sky_area,
            diff,
            num_points,
            diff_method,
            m_min,
            m_max,
            z_max,
            cosmo,
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_run_total_kappa_by_multiprocessing, args)
        average_kappa_list.append(results)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return average_kappa_list


def worker_run_total_kappa_by_multiprocessing(
    iter_num,
    sky_area,
    diff,
    num_points,
    diff_method,
    m_min,
    m_max,
    z_max,
    cosmo,
):
    """Computes the average kappa value for a given iteration, sky area,
    differential, number of points, differential method, minimum and maximum
    mass, maximum redshift, and cosmology.

    :param iter_num: Iteration number.
    :type iter_num: int
    :param sky_area: Sky area over which Halos are sampled.
    :type sky_area: float
    :param diff: Differential to use in the computation.
    :type diff: float
    :param num_points: Number of points to use in the computation.
    :type num_points: int
    :param diff_method: Method to use for the differential computation.
    :type diff_method: str
    :param m_min: Minimum halo mass.
    :type m_min: float
    :param m_max: Maximum halo mass.
    :type m_max: float
    :param z_max: Maximum redshift value in z_range.
    :type z_max: float
    :param cosmo: Cosmology model to use.
    :type cosmo: astropy.cosmology
    :return: Average kappa value.
    :rtype: float
    """

    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    nhalos_lens = HalosLensBase(
        halos_list=nhalos,
        sky_area=sky_area,
        cosmo=cosmo,
        z_source=z_max,
        mass_sheet=False,
    )

    _, total_kappa = nhalos_lens.halos_compute_kappa(
        diff=diff,
        num_points=num_points,
        diff_method=diff_method,
        enhance_pos=False,
    )
    average_kappa = np.mean(total_kappa)
    return average_kappa


def run_average_mass_by_multiprocessing(
    n_iterations=1,
    sky_area=0.0001,
    m_min=None,
    m_max=None,
    z_max=None,
):
    """Calculate the average mass of halos over multiple iterations using
    multiprocessing. This method was built for verify the
    expected_mass_at_redshift in halos.py.

    :param n_iterations: Number of iterations to run, defaults to 1
    :type n_iterations: int, optional
    :param sky_area: The sky area over which to calculate the halo mass, defaults to
        0.0001
    :type sky_area: float, optional
    :param m_min: The minimum mass of halos to consider, defaults to None
    :type m_min: float, optional
    :param m_max: The maximum mass of halos to consider, defaults to None
    :type m_max: float, optional
    :param z_max: The maximum redshift to consider, defaults to None. If None, uses 5.0
    :type z_max: float, optional
    :return: A list of average mass values for each iteration
    :rtype: list
    """
    if z_max is None:
        z_max = 5.0
        warnings.warn("No maximum redshift provided, using default 5.0")

    z_bins = np.arange(0, z_max + 0.05, 0.05)
    total_mass_sums = np.zeros(len(z_bins) - 1)

    start_time = time.time()

    args = [(i, sky_area, m_min, m_max, z_max) for i in range(n_iterations)]

    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_run_average_mass_by_multiprocessing, args)
        for result in results:
            total_mass_sums += (
                result  # Sum up the mass for each bin across all iterations
            )

    average_masses = total_mass_sums / n_iterations

    end_time = time.time()
    print(
        f"The {n_iterations} iterations took {(end_time - start_time):.2f} seconds to run"
    )
    return average_masses


def worker_run_average_mass_by_multiprocessing(
    iter_num,
    sky_area,
    m_min,
    m_max,
    z_max,
):
    """Worker function for Calculate the average mass of halos for each
    redshift bin.

    :param iter_num: The iteration number.
    :type iter_num: int
    :param sky_area: The sky area over which Halos are sampled.
    :type sky_area: float
    :param m_min: The minimum mass of the Halos to consider.
    :type m_min: float
    :param m_max: The maximum mass of the Halos to consider.
    :type m_max: float
    :param z_max: The maximum redshift of the Halos to consider.
    :type z_max: float
    :returns: The total mass of Halos for each redshift bin.
    :rtype: np.ndarray
    """
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    halos = npipeline.halos
    z_halos = halos["z"]
    mass_halos = halos["mass"]

    z_bins = np.arange(0, z_max + 0.05, 0.05)
    digitized = np.digitize(z_halos, z_bins) - 1
    bin_mass_sums = np.array(
        [
            mass_halos[digitized == i].sum() if np.any(digitized == i) else 0
            for i in range(len(z_bins) - 1)
        ]
    )

    return bin_mass_sums
