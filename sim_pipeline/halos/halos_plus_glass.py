import numpy as np
from sim_pipeline.Pipelines.halos_pipeline import HalosSkyPyPipeline
from sim_pipeline.halos.halos_lens import HalosLens
from astropy.units import Quantity
from tqdm import tqdm
import time
from scipy import stats
import warnings


def read_glass_data(file_name="kgdata.npy"):
    """
    This function loads a numpy data file and extracts the kappa and gamma values.

    Parameters:
        file_name (str, optional): The name of the file to load. Defaults to "kgdata.npy".

    Returns:
        tuple: A tuple containing two numpy arrays for kappa and gamma values.
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


def generate_kappa_gamma_using_halos(skypy_config=None, skyarea=0.0001, cosmo=None, samples_number_for_one_halos=1000,
                                     renders_numbers=500):
    """
    For given parameters,rendering halos to generate kappa and gamma values.

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


def generate_halos_multiple_times(n_times=20, skypy_config=None, skyarea=0.0001, cosmo=None,
                                  samples_number_for_one_halos=1000, renders_numbers=500):
    """
    For given parameters,rendering halos to generate kappa and gamma values multiple times.

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
    start_time = time.time()
    for _ in tqdm(range(n_times)):
        kappa_random_halos, gamma_random_halos = \
            generate_kappa_gamma_using_halos(skypy_config=skypy_config,
                                             skyarea=skyarea,
                                             cosmo=cosmo,
                                             samples_number_for_one_halos=samples_number_for_one_halos,
                                             renders_numbers=renders_numbers)
        accumulated_kappa_random_halos.extend(kappa_random_halos)
        accumulated_gamma_random_halos.extend(gamma_random_halos)

    print(f"Elapsed time: {time.time() - start_time} seconds")

    return np.array(accumulated_kappa_random_halos), np.array(accumulated_gamma_random_halos)


def halos_plus_glass(kappa_random_halos, gamma_random_halos, kappa_random_glass, gamma_random_glass):
    """
    This function combines the kappa and gamma values from the halos and Glass, and returns the combined kappa and
    gamma values.

    Parameters
    ----------
    kappa_random_halos (numpy.ndarray): The randomly resampled kappa values from the halos.
    gamma_random_halos (numpy.ndarray): The randomly resampled gamma values from the halos.
    kappa_random_glass (numpy.ndarray): The randomly resampled kappa values from Glass.
    gamma_random_glass (numpy.ndarray): The randomly resampled gamma values from Glass.

    Returns
    -------
    tuple: A tuple containing two numpy arrays for the combined kappa and gamma values.
    """
    total_kappa = [k_h + k_g for k_h, k_g in zip(kappa_random_halos, kappa_random_glass)]

    random_numbers = np.random.uniform(-1, 1, size=len(gamma_random_halos))
    total_gamma = [
        np.sqrt((g_h ** 2) + (g_g ** 2) + (2 * r * g_h * g_g))
        for g_h, g_g, r in zip(gamma_random_halos, gamma_random_glass, random_numbers)
    ]

    return total_kappa, total_gamma
    # TODO: Add math
