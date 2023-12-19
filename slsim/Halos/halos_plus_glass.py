import numpy as np
from slsim.Pipelines.halos_pipeline import HalosSkyPyPipeline
from slsim.Halos.halos_lens import HalosLens
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from tqdm.notebook import tqdm
import time
from scipy import stats
import warnings
from multiprocessing import get_context


def read_glass_data(file_name="kgdata.npy"):
    """This function loads a numpy data file and extracts the kappa and gamma
    vealues.

    Parameters:
        file_name (str, optional): The name of the file to load. Defaults to "kgdata.npy".

    Returns:
        tuple: A tuple containing two numpy arrays for kappa and gamma values and nside of the data.
    """

    def read_data_file(file_name):
        try:
            data = np.load(file_name)
            return data
        except FileNotFoundError:
            print(f"Error: {file_name} not found.")
            return None
        except Exception as e:
            print(f"Error occurred while loading {file_name}: {e}")
            return None

    data_array = read_data_file(file_name)
    kappa_values = data_array[:, 0]
    gamma_values = data_array[:, 1]
    nside = int(np.sqrt(len(kappa_values) / 12))
    return kappa_values, gamma_values, nside


def generate_samples_from_glass(kappa_values, gamma_values, n=10000):
    """This function fits a Gaussian Kernel Density Estimation (KDE) to the
    joint distribution of kappa and gamma values, and then generates a random
    sample from this distribution.

    Parameters
    ----------
        kappa_values (numpy.ndarray): The kappa values.
        gamma_values (numpy.ndarray): The gamma values.
        n (int, optional): The number of random numbers to generate. Defaults to 10000.

    Returns
    -------
        tuple: A tuple containing two numpy arrays for the randomly resampled kappa and gamma values.
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

    Parameters
    ----------
    nside (int): The HEALPix resolution parameter. The total number of pixels is 12 * nside ** 2.
    deg2 (bool, optional): If True, the sky area is returned in square degrees. If False, the sky area is returned
    in square arcseconds. Defaults to True.

    Returns
    -------
        float: The sky area per pixel in either degree^2 or arcsecond^2, depending on the value of deg2.
    """
    if deg2:
        skyarea = 129600 / (12 * np.pi * nside ** 2)  # sky area in square degrees
    else:
        skyarea = 1679616000000 / (
                12 * np.pi * nside ** 2
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
    """Given the specified parameters, this function repeatedly renders same
    halo list (same z & m) for different positions
    (`samples_number_for_one_halos`) times and then ensuring that the mean of
    the kappa values is zero, then get the kde of the weak-lensing distribution
    for this halo list and resample `renders_numbers` sets of the corresponding
    convergence (`kappa`) and shear (`gamma`) values for this halo list.

    Parameters
    ----------
    skypy_config: string or None
        path to SkyPy configuration yaml file. If None, the default skypy configuration file is used.
    skyarea : float, optional
        The sky area in square degrees. Default is 0.0001.
    cosmo : astropy.cosmology.FlatLambdaCDM, optional
        The cosmological model. If None, a default FlatLambdaCDM model with H0=70 and Om0=0.3 is used.
    m_min : float or None, optional
        Minimum halo mass for the pipeline. Default is None.
    m_max : float or None, optional
        Maximum halo mass for the pipeline. Default is None.
    z_max : float or None, optional
        Maximum redshift for the pipeline. Default is None.
    samples_number_for_one_halos : int, optional
        The number of samples for each halo. Default is 1000.
    renders_numbers : int, optional
        The number of random numbers to generate. Default is 500.

    Returns
    -------
    kappa_random_halos : numpy.ndarray
        The randomly resampled kappa values.
    gamma_random_halos : numpy.ndarray
        The randomly resampled gamma values.
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

    halos_lens = HalosLens(
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

    mean_kappa = np.mean(kappa_values_halos)
    modified_kappa_halos = kappa_values_halos - mean_kappa

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
    """Given the specified parameters, this function repeatedly renders same
    halo list (same z & m) for different positions
    (`samples_number_for_one_halos`) times and then ensuring that the mean of
    the kappa values is zero, then get the kde of the weak-lensing distribution
    for this halo list and resample `renders_numbers` sets of the corresponding
    convergence (`kappa`) and shear (`gamma`) values for this halo list. This
    process is repeated `n_times` to accumulate the results.

    Parameters
    ----------
    n_times: int, optional
        The number of times to repeat the generation of kappa and gamma values. Default is 20.
    skypy_config: string or None
        path to SkyPy configuration yaml file. If None, the default skypy configuration file is used.
    skyarea : float, optional
        The sky area in square degrees. Default is 0.0001.
    cosmo : astropy.cosmology.FlatLambdaCDM, optional
        The cosmological model. If None, a default FlatLambdaCDM model with H0=70 and Om0=0.3 is used.
    m_min : float or None, optional
        Minimum halo mass for the pipeline. Default is None.
    m_max : float or None, optional
        Maximum halo mass for the pipeline. Default is None.
    z_max : float or None, optional
        Maximum redshift for the pipeline. Default is None.
    samples_number_for_one_halos : int, optional
        The number of samples for each halo. Default is 1000.
    renders_numbers : int, optional
        The number of random numbers to generate. Default is 500.

    Returns
    -------
    kappa_random_halos : numpy.ndarray
        The accumulated randomly resampled kappa values.
    gamma_random_halos : numpy.ndarray
        The accumulated randomly resampled gamma values.
    """
    accumulated_kappa_random_halos = []
    accumulated_gamma_random_halos = []
    n_times_range = range(n_times)
    if n_times > 100:
        n_times_range = tqdm(n_times_range, desc="Processing iterations")

    start_time = time.time()
    for _ in n_times_range:
        kappa_random_halos, gamma_random_halos = generate_maps_kmean_zero_using_halos(
            skypy_config=skypy_config,
            skyarea=skyarea,
            cosmo=cosmo,
            samples_number_for_one_halos=samples_number_for_one_halos,
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

    Parameters
    ----------
    kappa_random_halos : numpy.ndarray
        The randomly resampled kappa values from the Halos.
    gamma_random_halos : numpy.ndarray
        The randomly resampled gamma values from the Halos.
    kappa_random_glass : numpy.ndarray
        The randomly resampled kappa values from Glass.
    gamma_random_glass : numpy.ndarray
        The randomly resampled gamma values from Glass.

    Returns
    -------
    tuple
        A tuple containing two list of numpy arrays for the combined kappa and gamma values.

    Notes
    -----
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
        np.sqrt((g_h ** 2) + (g_g ** 2) + (2 * r * g_h * g_g))
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
        z_max=None,
        mass_sheet_correction=True,
        listmean=False,
):
    """Under the specified `sky_area`, generate `n_iterations` sets of halo
    lists. For each set, simulate `samples_number` times to obtain the
    convergence (`kappa`) and shear (`gamma`) values at the origin. These
    values are directly appended without any additional processing (i.e.,
    without KDE computation, resampling, or subtracting the mean kappa value).

    Parameters
    ----------
    n_iterations : int, optional
        Number of iterations or halo lists to generate. Defaults to 1.
    sky_area : float, optional
        Total sky area (in steradians) over which Halos are distributed. Defaults to 0.0001 steradians.
    samples_number : int, optional
        Number of samples for statistical calculations. Defaults to 1500.
    cosmo : astropy.cosmology instance, optional
        Cosmology used for the simulations. If not provided, the default astropy cosmology is used.
    m_min : float, optional
        Minimum mass of the Halos to consider. If not provided, no lower limit is set.
    m_max : float, optional
        Maximum mass of the Halos to consider. If not provided, no upper limit is set.
    z_max : float, optional
        Maximum redshift of the Halos to consider. If not provided, no upper limit is set.

    Returns
    -------
    kappa_values_total : list
        Combined list of convergence (`kappa`) values from all iterations.
    gamma_values_total : list
        Combined list of shear (`gamma`) values from all iterations.

    Notes
    -----
    This function initializes a halo distribution pipeline for each iteration, simulates Halos,
    and then computes the lensing properties (`kappa` and `gamma`). The results from all iterations
    are concatenated to form the final output lists.

    Warnings
    --------
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
    if n_iterations > 30:
        iter_range = tqdm(iter_range, desc="Processing halo-lists iterations")

    start_time = time.time()  # Note the start time
    for _ in iter_range:
        npipeline = HalosSkyPyPipeline(
            sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
        )
        nhalos = npipeline.halos
        if mass_sheet_correction:
            mass_sheet_correction_list = npipeline.mass_sheet_correction
            nhalos_lens = HalosLens(
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
            nhalos_lens = HalosLens(
                halos_list=nhalos,
                sky_area=sky_area,
                cosmo=cosmo,
                samples_number=samples_number,
                mass_sheet=False,
                z_source=z_max
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
):
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos
    if mass_sheet_correction:
        mass_sheet_correction_list = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=mass_sheet_correction_list,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
        )
    nkappa_gamma_distribution = (
        nhalos_lens.get_kappa_gamma_distib_without_multiprocessing(gamma_tot=True, listmean=listmean)
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
        z_max=None,
        mass_sheet_correction=True,
        listmean=False,
):
    """Under the specified `sky_area`, generate `n_iterations` sets of halo
    lists. For each set, simulate `samples_number` times to obtain the
    convergence (`kappa`) and shear (`gamma`) values at the origin. These
    values are directly appended without any additional processing (i.e.,
    without KDE computation, resampling, or subtracting the mean kappa value),
    the process procedure `n_iterations` generate with multiprocessing.

    Parameters
    ----------
    listmean
    n_iterations : int, optional
        Number of iterations or halo lists to generate. Defaults to 1.
    sky_area : float, optional
        Total sky area (in steradians) over which Halos are distributed. Defaults to 0.0001 steradians.
    samples_number : int, optional
        Number of samples for statistical calculations. Defaults to 1500.
    cosmo : astropy.cosmology instance, optional
        Cosmology used for the simulations. If not provided, the default astropy cosmology is used.
    m_min : float, optional
        Minimum mass of the Halos to consider. If not provided, no lower limit is set.
    m_max : float, optional
        Maximum mass of the Halos to consider. If not provided, no upper limit is set.
    z_max : float, optional
        Maximum redshift of the Halos to consider. If not provided, no upper limit is set.
    mass_sheet_correction : bool, optional
        If True, apply mass sheet correction. Defaults to True.


    Returns
    -------
    kappa_values_total : list
        Combined list of convergence (`kappa`) values from all iterations.
    gamma_values_total : list
        Combined list of shear (`gamma`) values from all iterations.

    Notes
    -----
    This function initializes a halo distribution pipeline for each iteration, simulates Halos,
    and then computes the lensing properties (`kappa` and `gamma`). The results from all iterations
    are concatenated to form the final output lists.

    Warnings
    --------
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
        (i, sky_area, m_min, m_max, z_max, cosmo, samples_number, mass_sheet_correction, listmean)
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
        z_max=None,
        mass_sheet_correction=True,
        listmean=False,
        output_format="dict",
):
    """Run the kappa-gamma external convergence distribution for a given number
    of iterations using multiprocessing.

    This function generates kappa and gamma distributions using the provided parameters and
    a worker function (`worker_kappaext_gammaext_kde`). The distributions are generated
    in parallel over the specified number of iterations.

    Parameters
    ----------
    n_iterations : int, optional
        The number of iterations to run the simulation. Defaults to 1.

    sky_area : float, optional
        The area of the sky under consideration (in steradians). Defaults to 0.0001.

    samples_number : int, optional
        The number of samples to be used. Defaults to 1.

    cosmo : astropy.cosmology instance, optional
        The cosmology to be used. If not provided, it defaults to FlatLambdaCDM with H0=70 and Om0=0.3.

    m_min : float, optional
        The minimum mass for the simulation. If None, it is decided by the worker function.

    m_max : float, optional
        The maximum mass for the simulation. If None, it is decided by the worker function.

    z_max : float, optional
        The maximum redshift for the simulation. If None, it is decided by the worker function.

    mass_sheet_correction : bool, optional
        If True, apply mass sheet correction. Defaults to True.

    output_format : str, optional
        The format in which the results should be returned, either as 'dict' or 'vector'. Defaults to 'dict'.

    Returns
    -------
    list
        A list of kappa and gamma values across all iterations.

    Notes
    -----
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

    Parameters
    ----------
    iter_num : int
        The iteration number, mainly used for tracking in a parallel processing context.

    sky_area : float
        The area of the sky under consideration (in steradians).

    m_min : float
        The minimum mass for the simulation.

    m_max : float
        The maximum mass for the simulation.

    z_max : float
        The maximum redshift for the simulation.

    cosmo : astropy.Cosmology instance
        The cosmology to be used.

    samples_number : int
        The number of samples to be used.

    mass_sheet_correction : bool
        If True, apply mass sheet correction.

    output_format : str, optional
        The format in which the results should be returned, either as 'dict' or 'vector'. Defaults to 'dict'.

    Returns
    -------
    list
        A list of kappa and gamma values for the specified parameters.

    Notes
    -----
    This function is primarily intended to be used as a worker function in a parallel processing
    framework, where multiple instances of the function can be run simultaneously.
    """
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
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
):
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if zs is None:
        zs = 1.5
        warnings.warn(
            "No source redshift provided, instead uses 1.5"
        )
    if zd is None:
        zd = 1.0
        warnings.warn(
            "No lens redshift provided, instead uses 1.0"
        )
    if z_max is None:
        z_max = 5.0
        warnings.warn(
            "No maximum redshift provided, instead uses 5.0"
        )

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
):
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
            mass_sheet=True,
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
        )
    distributions = nhalos_lens.get_kappaext_gammaext_distib_zdzs(zd, zs, listmean=listmean)
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
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if zs is None:
        zs = 1.5
        warnings.warn(
            "No source redshift provided, instead uses 1.5"
        )
    if zd is None:
        zd = 1.0
        warnings.warn(
            "No lens redshift provided, instead uses 1.0"
        )
    if z_max is None:
        z_max = 5.0
        warnings.warn(
            "No maximum redshift provided, instead uses 5.0"
        )

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
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
        )
    distributions, lensinstance = nhalos_lens.get_alot_distib_(zd, zs)
    return distributions, lensinstance


def run_compute_kappa_in_bins_by_multiprocessing(
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
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if zs is None:
        zs = 1.5
        warnings.warn(
            "No source redshift provided, instead uses 1.5"
        )
    if zd is None:
        zd = 1.0
        warnings.warn(
            "No lens redshift provided, instead uses 1.0"
        )
    if z_max is None:
        z_max = 5.0
        warnings.warn(
            "No maximum redshift provided, instead uses 5.0"
        )

    kappa_dict_tot = []

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
        results = pool.starmap(worker_compute_kappa_in_bins, args)
        kappa_dict_tot.extend(results)

    kappa_dict_tot = [item for sublist in kappa_dict_tot for item in sublist]

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return kappa_dict_tot


def worker_compute_kappa_in_bins(
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
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
        )
    kappa_dict = nhalos_lens.compute_kappa_in_bins()
    return kappa_dict


def run_azimuthal_average_by_multiprocessing(
        cosmo,
        m_min,
        m_max,
        z_max,
        mass_sheet_correction,
        n_iterations=1,
        sky_area=0.0001,
        samples_number=1,
):
    azimuthal_dict_tot = []

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
        )
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_azimuthal_average, args)
        azimuthal_dict_tot.extend(results)

    azimuthal_dict_tot = [item for sublist in azimuthal_dict_tot for item in sublist]

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return azimuthal_dict_tot


def worker_azimuthal_average(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        mass_sheet_correction
):
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    if mass_sheet_correction:
        nmass_sheet_correction = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=nmass_sheet_correction,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
        )
    azimuthal_dict = nhalos_lens.azimuthal_average_kappa_dict()
    return azimuthal_dict


def run_total_mass_by_multiprocessing(
        n_iterations=1,
        sky_area=0.0001,
        cosmo=None,
        m_min=None,
        m_max=None,
        z_max=None,
):
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if z_max is None:
        z_max = 5.0
        warnings.warn(
            "No maximum redshift provided, instead uses 5.0"
        )

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
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    nhalos_lens = HalosLens(
        halos_list=nhalos,
        sky_area=sky_area,
        cosmo=cosmo,
        samples_number=1,
        z_source=z_max,
        mass_sheet=False
    )

    total_mass = nhalos_lens.total_halo_mass()
    return total_mass


def worker_run_kappa_mean_range(
        iter_num,
        sky_area,
        m_min,
        m_max,
        z_max,
        cosmo,
        samples_number,
        mass_sheet_correction,
        diff
):
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos
    if mass_sheet_correction:
        mass_sheet_correction_list = npipeline.mass_sheet_correction
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            mass_correction_list=mass_sheet_correction_list,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            z_source=z_max,
        )
    else:
        nhalos_lens = HalosLens(
            halos_list=nhalos,
            sky_area=sky_area,
            cosmo=cosmo,
            samples_number=samples_number,
            mass_sheet=False,
            z_source=z_max
        )
    kappa_mean, kappa_2sigma, mass, mass_divide_kcrit_tot = (
        nhalos_lens.get_kappa_mass_relation(diff=diff)
    )
    return kappa_mean, kappa_2sigma, mass, mass_divide_kcrit_tot


def run_kappa_mean_range_by_multiprocessing(
        n_iterations=1,
        sky_area=0.0001,
        samples_number=1500,
        cosmo=None,
        m_min=None,
        m_max=None,
        z_max=None,
        mass_sheet_correction=True,
        diff=1.0,
):
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        import astropy.cosmology
        cosmo = astropy.cosmology.default_cosmology.get()
    mean_kappa_total = []
    two_sigma_total = []
    mass_total = []
    mass_divide_kcrit_total = []

    start_time = time.time()  # Note the start time

    args = [
        (i, sky_area, m_min, m_max, z_max, cosmo, samples_number, mass_sheet_correction, diff)
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_run_kappa_mean_range, args)

    for mean_kappa, two_sigma, mass, mass_divide_kcrit_tot in results:
        mean_kappa_total.append(mean_kappa)
        two_sigma_total.append(two_sigma)
        mass_total.append(mass)
        mass_divide_kcrit_total.append(mass_divide_kcrit_tot)

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return mean_kappa_total, two_sigma_total, mass_total, mass_divide_kcrit_total


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
    if cosmo is None:
        warnings.warn(
            "No cosmology provided, instead uses astropy.cosmology default cosmology"
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    if z_max is None:
        z_max = 5.0
        warnings.warn(
            "No maximum redshift provided, instead uses 5.0"
        )

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
    npipeline = HalosSkyPyPipeline(
        sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max
    )
    nhalos = npipeline.halos

    nhalos_lens = HalosLens(
        halos_list=nhalos,
        sky_area=sky_area,
        cosmo=cosmo,
        samples_number=1,
        z_source=z_max,
        mass_sheet=False
    )

    _, total_kappa = nhalos_lens.compute_kappa(diff=diff,
                                               num_points=num_points,
                                               diff_method=diff_method,
                                               enhance_pos=False, )
    average_kappa = np.mean(total_kappa)
    return average_kappa


def compute_total_kappa_for_sky_area(cosmo,
                                     n_iterations=200,
                                     diff=0.0000001,
                                     num_points=50,
                                     z_max=None,
                                     m_min=None,
                                     m_max=None):
    sky_areas = np.arange(0.0001, 0.00105, 0.00005)
    average_kappa = {}

    for sky_area in sky_areas:
        average_kappa_list = run_total_kappa_by_multiprocessing(n_iterations=n_iterations,
                                                                sky_area=sky_area,
                                                                diff=diff,
                                                                num_points=num_points,
                                                                diff_method="square",
                                                                cosmo=cosmo,
                                                                m_min=m_min,
                                                                m_max=m_max,
                                                                z_max=z_max)
        average_kappa[sky_area] = average_kappa_list
    return average_kappa


def worker_run_halos(iter_num, sky_area, m_min_str, m_max_str, m_min, m_max, z_max):
    npipeline = HalosSkyPyPipeline(sky_area=sky_area, m_min=m_min_str, m_max=m_max_str, z_max=z_max)

    nhalos = npipeline.halos
    halos_mass = nhalos['mass']
    halos_z = nhalos['z']
    assert len(halos_mass) == len(halos_z)
    halos_mass_list = [item[0] for item in halos_mass]
    return halos_mass_list, halos_z


def run_halos_by_multiprocessing(
        m_min_str,
        m_max_str,
        m_min,
        m_max,
        z_max,
        n_iterations=1,
        sky_area=0.0001,
):
    halos_mass_total = []
    halos_z_total = []
    number_total = []

    start_time = time.time()  # Note the start time

    args = [
        (i, sky_area, m_min_str, m_max_str, m_min, m_max, z_max)
        for i in range(n_iterations)
    ]

    # Use multiprocessing
    with get_context("spawn").Pool() as pool:
        results = pool.starmap(worker_run_halos, args)

    for halos_mass, halos_z in results:
        halos_mass_total.extend(halos_mass)
        halos_z_total.extend(halos_z)
        number_total.append(len(halos_mass))

    end_time = time.time()  # Note the end time
    print(
        f"The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run"
    )
    return halos_mass_total, halos_z_total, number_total
