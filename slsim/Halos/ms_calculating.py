from slsim.Halos.read_millennium import ReadMS
from slsim.Halos.ms_halos_lens import HalosMSLens
from astropy.cosmology import default_cosmology
import multiprocessing
import os
import glob
from tqdm.notebook import tqdm


def calculate_kappa_gamma(file_path=None,
                          selecting_area=0.00082,
                          z_source=5,
                          cosmo=None,
                          sample_size=1,
                          gamma12=False,
                          diff=0.000001,
                          diff_method="square"):
    # Set the cosmology
    if cosmo is None:
        cosmo = default_cosmology.get()

    # Initialize ReadMS and get tables
    read_ms = ReadMS(file_path=file_path, selecting_area=selecting_area, z_source=z_source, cosmo=cosmo,
                     sample_size=sample_size)
    tables = read_ms.get_tables()

    # Assuming you want to iterate through all tables
    all_kappa = []
    all_gamma = []
    if gamma12:
        print('only gamma12 false now')
    for halos in tables:
        # Initialize HalosMSLens with filtered halos
        a_halos_MSLens = HalosMSLens(halos_list=halos, cosmo=cosmo, sky_area=selecting_area, z_source=z_source)
        kappa, gamma = a_halos_MSLens.get_convergence_shear(gamma12=False, diff=diff, diff_method=diff_method)
        # Collect results
        all_kappa.append(kappa)
        all_gamma.append(gamma)

    return all_kappa, all_gamma


def worker_process_table(halos, cosmo, selecting_area, z_source, diff, diff_method):
    a_halos_MSLens = HalosMSLens(halos_list=halos, cosmo=cosmo, sky_area=selecting_area, z_source=z_source)
    kappa, gamma = a_halos_MSLens.get_convergence_shear(gamma12=False, diff=diff, diff_method=diff_method)
    return kappa, gamma


def calculate_kappa_gamma_with_muiltprocessing(file_path=None,
                                               selecting_area=0.00082,
                                               z_source=5,
                                               cosmo=None,
                                               sample_size=1,
                                               gamma12=False,
                                               diff=0.000001,
                                               diff_method="square"):
    if cosmo is None:
        cosmo = default_cosmology.get()

    read_ms = ReadMS(file_path=file_path,
                     selecting_area=selecting_area,
                     z_source=z_source, cosmo=cosmo,
                     sample_size=sample_size)
    tables = read_ms.get_tables()

    # Using multiprocessing
    pool = multiprocessing.Pool(processes=5)
    results = [pool.apply_async(worker_process_table,
                                args=(halos, cosmo, selecting_area, z_source, diff, diff_method)) for halos in tables]

    # Close the pool and wait for each task to complete
    pool.close()
    pool.join()

    # Extract results
    all_kappa = [result.get()[0] for result in results]
    all_gamma = [result.get()[1] for result in results]

    return all_kappa, all_gamma


def worker_halos_table(halos, z_source, mass_cut=None):
    halos_list = halos[halos['z'] <= z_source]
    if mass_cut is not None:
        halos_list = halos_list[halos_list['mass'] >= mass_cut]
    mass_list = halos_list["mass"]
    z_list = halos_list["z"]
    assert len(mass_list) == len(z_list)
    return mass_list, z_list


def get_halos_mass_with_muiltprocessing(file_path=None,
                                        selecting_area=0.00082,
                                        z_source=5,
                                        cosmo=None,
                                        mass_cut=None,
                                        sample_size=1):
    if cosmo is None:
        cosmo = default_cosmology.get()

    read_ms = ReadMS(file_path=file_path,
                     selecting_area=selecting_area,
                     z_source=z_source,
                     cosmo=cosmo,
                     sample_size=sample_size)
    tables = read_ms.get_tables()

    # Using multiprocessing with 'with' statement
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(worker_halos_table, args=(halos, z_source, mass_cut)) for halos in tables]

        # Extract results within the 'with' block
        all_masses = []
        all_zs = []
        all_lengths = []
        for result in results:
            mass_list, z_list = result.get()
            all_masses.extend(mass_list)
            all_zs.extend(z_list)
            all_lengths.append(len(mass_list))

    return all_masses, all_zs, all_lengths

def kappa_gamma_from_files(file_path=None,
                           selecting_area=0.00082,
                           z_source=5,
                           cosmo=None,
                           sample_size=1,
                           gamma12=False,
                           diff=0.000001,
                           diff_method="square"):
    if cosmo is None:
        cosmo = default_cosmology.get()

    if file_path is None or not os.path.isdir(file_path):
        raise ValueError("Invalid file path provided")

    # Find all text files in the specified path
    txt_files = glob.glob(os.path.join(file_path, '*.txt'))

    all_kappa_results = []
    all_gamma_results = []

    # Process each file
    for txt_file in tqdm(txt_files, desc="Processing files"):
        kappas, gammas = calculate_kappa_gamma_with_muiltprocessing(file_path=txt_file,
                                                                    selecting_area=selecting_area,
                                                                    z_source=z_source,
                                                                    cosmo=cosmo,
                                                                    sample_size=sample_size,
                                                                    gamma12=gamma12,
                                                                    diff=diff,
                                                                    diff_method=diff_method)
        all_kappa_results.extend(kappas)
        all_gamma_results.extend(gammas)

    return all_kappa_results, all_gamma_results


def get_mass_z_from_files(file_path=None,
                          selecting_area=0.00082,
                          z_source=5,
                          cosmo=None,
                          mass_cut=None,
                          sample_size=1):
    if cosmo is None:
        cosmo = default_cosmology.get()

    # Check if file_path is provided and is a valid directory
    if file_path is None or not os.path.isdir(file_path):
        raise ValueError("Invalid file path provided")

    # Find all text files in the specified path
    txt_files = glob.glob(os.path.join(file_path, '*.txt'))

    # Prepare lists to collect results
    all_masses_results = []
    all_zs_results = []
    all_lengths_results = []

    # Process each file
    for txt_file in tqdm(txt_files, desc="Processing files"):
        masses, zs, lengths = get_halos_mass_with_muiltprocessing(file_path=txt_file,
                                                                  selecting_area=selecting_area,
                                                                  z_source=z_source,
                                                                  cosmo=cosmo,
                                                                  mass_cut=mass_cut,
                                                                  sample_size=sample_size)
        all_masses_results.extend(masses)
        all_zs_results.extend(zs)
        all_lengths_results.extend(lengths)

    return all_masses_results, all_zs_results, all_lengths_results