import numpy as np
from sim_pipeline.Pipelines.halos_pipeline import HalosSkyPyPipeline
from sim_pipeline.halos.halos_lens import HalosLens
from astropy.cosmology import FlatLambdaCDM
from tqdm.notebook import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import starmap
import time
from scipy import stats
import warnings
import multiprocessing
from multiprocessing import get_context


def read_glass_data(file_name="kgdata.npy"):
    """
    This function loads a numpy data file and extracts the kappa and gamma values.

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
    """
    This function fits a Gaussian Kernel Density Estimation (KDE) to the joint distribution of kappa and gamma values,
    and then generates a random sample from this distribution.

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
    """
    This function calculates the sky area corresponding to each pixel when the sky is divided into 12 * nside ** 2
    equal areas. 4*pi*(180/pi)^2=129600/pi deg^2; 4*pi*(180/pi)^2*(3600)^2=1679616000000/pi arcsec^2.

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
        skyarea = 1679616000000 / (12 * np.pi * nside ** 2)  # sky area in square arcseconds
    return skyarea


def generate_maps_kmean_zero_using_halos(skypy_config=None, skyarea=0.0001, cosmo=None,
                                         samples_number_for_one_halos=1000,
                                         renders_numbers=500):
    """
    Given the specified parameters, this function repeatedly renders same halo list (same z & m) for different
    positions (`samples_number_for_one_halos`) times and then ensuring that the mean of the kappa values is zero,
    then get the kde of the weak-lensing distribution for this halo list and resample `renders_numbers` sets of the
    corresponding convergence (`kappa`) and shear (`gamma`) values for this halo list.

    Parameters
    ----------
    skypy_config: string or None
        path to SkyPy configuration yaml file. If None, the default skypy configuration file is used.
    skyarea : float, optional
        The sky area in square degrees. Default is 0.0001.
    cosmo : astropy.cosmology.FlatLambdaCDM, optional
        The cosmological model. If None, a default FlatLambdaCDM model with H0=70 and Om0=0.3 is used.
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
        warnings.warn("No cosmology provided, instead uses astropy.cosmology default cosmology")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    sky_area = skyarea

    pipeline = HalosSkyPyPipeline(sky_area=sky_area, skypy_config=skypy_config)
    halos = pipeline.halos

    halos_lens = HalosLens(halos_list=halos, sky_area=sky_area, cosmo=cosmo,
                           samples_number=samples_number_for_one_halos)
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


def generate_meanzero_halos_multiple_times(n_times=20, skypy_config=None, skyarea=0.0001, cosmo=None,
                                           samples_number_for_one_halos=1000, renders_numbers=500):
    """
    Given the specified parameters, this function repeatedly renders same halo list (same z & m) for different
    positions (`samples_number_for_one_halos`) times and then ensuring that the mean of the kappa values is zero,
    then get the kde of the weak-lensing distribution for this halo list and resample `renders_numbers` sets of the
    corresponding convergence (`kappa`) and shear (`gamma`) values for this halo list. This process is repeated
    `n_times` to accumulate the results.

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
        kappa_random_halos, gamma_random_halos = \
            generate_maps_kmean_zero_using_halos(skypy_config=skypy_config,
                                                 skyarea=skyarea,
                                                 cosmo=cosmo,
                                                 samples_number_for_one_halos=samples_number_for_one_halos,
                                                 renders_numbers=renders_numbers)
        accumulated_kappa_random_halos.extend(kappa_random_halos)
        accumulated_gamma_random_halos.extend(gamma_random_halos)

    end_time = time.time()  # Note the end time
    print(f"Elapsed time: {end_time - start_time} seconds")

    return np.array(accumulated_kappa_random_halos), np.array(accumulated_gamma_random_halos)


def halos_plus_glass(kappa_random_halos, gamma_random_halos, kappa_random_glass, gamma_random_glass):
    r"""
    Combine the kappa and gamma values from halos and Glass distributions, returning the combined
    values. For halos kappa, it is suggested to use modified kappa values (i.e., direct kappa minus mean(kappa)).

    Parameters
    ----------
    kappa_random_halos : numpy.ndarray
        The randomly resampled kappa values from the halos.
    gamma_random_halos : numpy.ndarray
        The randomly resampled gamma values from the halos.
    kappa_random_glass : numpy.ndarray
        The randomly resampled kappa values from Glass.
    gamma_random_glass : numpy.ndarray
        The randomly resampled gamma values from Glass.

    Returns
    -------
    tuple
        A tuple containing two numpy arrays for the combined kappa and gamma values.

    Notes
    -----
    The total shear, :math:`\gamma_{\text{tot}}`, is computed based on the relationships:

    .. math:: \gamma_{\text{tot}} = \sqrt{(\gamma1_{\text{halos}}  + \gamma1_{\text{GLASS}})^2 + (\gamma2_{\text{
        halos}}  + \gamma2_{\text{GLASS}})^2}

    and

    .. math:: \gamma_{\text{tot}} = \sqrt{\gamma_{\text{halos}}^2 + \gamma_{\text{GLASS}}^2 + 2\gamma_{\text{
        halos}}\gamma_{\text{GLASS}} \times \cos(\alpha-\beta)}

    Given that :math:`\alpha` and :math:`\beta` are randomly distributed, and their difference, :math:`\alpha-\beta`,
    follows a normal distribution, the shear is given by:

    .. math:: \gamma_{\text{tot}}^2 = \gamma_{\text{halos}}^2 + \gamma_{\text{GLASS}}^2 + 2\gamma_{\text{
        halos}}\gamma_{\text{GLASS}} \cdot \text{random}(-1,1)
    """
    total_kappa = [k_h + k_g for k_h, k_g in zip(kappa_random_halos, kappa_random_glass)]

    random_numbers = np.random.uniform(-1, 1, size=len(gamma_random_halos))
    total_gamma = [
        np.sqrt((g_h ** 2) + (g_g ** 2) + (2 * r * g_h * g_g))
        for g_h, g_g, r in zip(gamma_random_halos, gamma_random_glass, random_numbers)
    ]

    return total_kappa, total_gamma
    # TODO: Add math


def run_halos_without_kde(n_iterations=1, sky_area=0.0001, samples_number=1500, cosmo=None, m_min=None, m_max=None,
                          z_max=None):
    """
    Under the specified `sky_area`, generate `n_iterations` sets of halo lists. For each set,
    simulate `samples_number` times to obtain the convergence (`kappa`) and shear (`gamma`) values
    at the origin. These values are directly appended without any additional processing (i.e., without
    KDE computation, resampling, or subtracting the mean kappa value).

    Parameters
    ----------
    n_iterations : int, optional
        Number of iterations or halo lists to generate. Defaults to 1.
    sky_area : float, optional
        Total sky area (in steradians) over which halos are distributed. Defaults to 0.0001 steradians.
    samples_number : int, optional
        Number of samples for statistical calculations. Defaults to 1500.
    cosmo : astropy.cosmology instance, optional
        Cosmology used for the simulations. If not provided, the default astropy cosmology is used.
    m_min : float, optional
        Minimum mass of the halos to consider. If not provided, no lower limit is set.
    m_max : float, optional
        Maximum mass of the halos to consider. If not provided, no upper limit is set.
    z_max : float, optional
        Maximum redshift of the halos to consider. If not provided, no upper limit is set.

    Returns
    -------
    kappa_values_total : list
        Combined list of convergence (`kappa`) values from all iterations.
    gamma_values_total : list
        Combined list of shear (`gamma`) values from all iterations.

    Notes
    -----
    This function initializes a halo distribution pipeline for each iteration, simulates halos,
    and then computes the lensing properties (`kappa` and `gamma`). The results from all iterations
    are concatenated to form the final output lists.

    Warnings
    --------
    If no cosmology is provided, the function uses the default astropy cosmology, which is a flat
    Lambda-CDM model with H0=70 and Om0=0.3.
    """
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM
        warnings.warn("No cosmology provided, instead uses astropy.cosmology default cosmology")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kappa_values_total = []
    gamma_values_total = []
    # Show progress only when n_iterations > 30
    iter_range = range(n_iterations)
    if n_iterations > 30:
        iter_range = tqdm(iter_range, desc="Processing halo-lists iterations")

    start_time = time.time()  # Note the start time
    for _ in iter_range:
        # Initialize the pipeline and get the halo list
        npipeline = HalosSkyPyPipeline(sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max)
        nhalos = npipeline.halos

        nhalos_lens = HalosLens(halos_list=nhalos, sky_area=sky_area, cosmo=cosmo, samples_number=samples_number)

        nkappa_gamma_distribution = nhalos_lens.get_kappa_gamma_distib(gamma_tot=True)
        nkappa_gamma_distribution = np.array(nkappa_gamma_distribution)  # Convert list of lists to numpy array

        nkappa_values_halos = nkappa_gamma_distribution[:, 0]
        ngamma_values_halos = nkappa_gamma_distribution[:, 1]

        kappa_values_total.extend(nkappa_values_halos)
        gamma_values_total.extend(ngamma_values_halos)

    end_time = time.time()  # Note the end time
    print(f'The {n_iterations} halo-lists took {(end_time - start_time)} seconds to run')

    return kappa_values_total, gamma_values_total


"""
def worker_run_halos_without_kde(iter_num, sky_area, m_min, m_max, z_max, cosmo, samples_number):

    npipeline = HalosSkyPyPipeline(sky_area=sky_area, m_min=m_min, m_max=m_max, z_max=z_max)
    nhalos = npipeline.halos

    nhalos_lens = HalosLens(halos_list=nhalos, sky_area=sky_area, cosmo=cosmo, samples_number=samples_number)

    nkappa_gamma_distribution = nhalos_lens.get_kappa_gamma_distib(gamma_tot=True)
    nkappa_gamma_distribution = np.array(nkappa_gamma_distribution)

    nkappa_values_halos = nkappa_gamma_distribution[:, 0]
    ngamma_values_halos = nkappa_gamma_distribution[:, 1]

    return nkappa_values_halos, ngamma_values_halos


def run_halos_without_kde(n_iterations=1, sky_area=0.0001, samples_number=1500, cosmo=None, m_min=None, m_max=None,
                          z_max=None):

    if cosmo is None:
        warnings.warn("No cosmology provided, instead uses astropy.cosmology default cosmology")
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kappa_values_total = []
    gamma_values_total = []

    args = [(i, sky_area, m_min, m_max, z_max, cosmo, samples_number) for i in range(n_iterations)]

    # Use multiprocessing
    with get_context('spawn').Pool() as pool:
        results = pool.starmap(worker_run_halos_without_kde, args)

    for nkappa, ngamma in results:
        kappa_values_total.extend(nkappa)
        gamma_values_total.extend(ngamma)

    return kappa_values_total, gamma_values_total

"""
# TODO: make the mulitprocessing work (mulit in mulit)
