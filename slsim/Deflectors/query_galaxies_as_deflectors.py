#!/usr/bin/env python
from math import *
import numpy as np
import srcprop as srcprop
import srclensprop as slp
from astropy.table import Table
from slsim.Deflectors.light2mass import get_velocity_dispersion
from uncertainties import unumpy
from astropy.cosmology import FlatLambdaCDM

##########################################################################################################
##  This script takes as input a foreground galaxy catalog with RA, DEC, redshift,                      ##
##  magnitudes, errors, position angle and the ellipticity and uses this information                    ##
##  to determine which of the galaxies will act as potential lenses, and returns an                     ##
##  output catalogs of sorted deflectors with the same parameters as in input catalog.                  ##
##                                                                                                      ##
##  Note that background source number density is artificially increased to                             ##
##  increase the chances of lensing due to the respective foreground galaxy.                            ##

##    To use the script in series and parallel:                                                         ##
##    -- Check the notebook 'example_query_galaxies_as_deflectors.ipynb'                                ##
##########################################################################################################


def find_potential_lenses(
    foreground_table,
    background_source,
    kwargs_deflector={
        "min_z": 0.2,
        "max_z": 1.9,
        "min_shear": 0.001,
        "max_shear": 0.02,
        "min_PA_shear": 0.0,
        "max_PA_shear": 180.0,
    },
    kwargs_source={
        "type": "galaxy",
        "min_z": 0.2,
        "max_z": 5.0,
        "min_mag": 25.0,
        "max_mag": 28.0,
        "boost_csect": 3,
        "ell_min": 0.1,
        "ell_max": 0.8,
        "PA_min": 0.0,
        "PA_max": 180.0,
        "R_Einst_min": 0.5,
        "R_Einst_max": 3.0,
    },
    cosmo=FlatLambdaCDM(H0=72, Om0=0.26),
    constants={"G": 4.2994e-9, "light_speed": 299792.458},
    parallel=False,
    save_source_file=False,
    chunk_offset=0,
):
    """This function identifies potential lensing galaxies (deflectors) from a
    foreground galaxy catalog, on the basis of their angular cross section, source
    luminosity function and their magnitudes and redshift falling within a certain bin,
    defined in the source and deflector kwargs.

    params:

    -- foreground_catalog: the foreground galaxy catalog with atleast the following params:
       redshift, gmag, rmag, imag, err-g, err-r, err-i, ellipticity
    type: astropy table
    NOTE: the foreground catalog should have the above column names, else, change the column names as shown in
          the notebook 'example_query_galaxies_as_deflectors.py'

    -- background_source: background source for the lensing, either galaxy or quasar
    type: string


    -- kwargs_source:  a dictionary containing all the background source parameters,
                     should have the following params:
                     {'type':'galaxy', 'min_z':3.0, 'max_z':6.0, 'min_mag':25.0, 'max_mag':32.0,
                     'boost_csect':3, 'ell_min':0.1, 'ell_max': 0.8, 'PA_min': 0., 'PA_max':180.,
                     'R_Einst_min': 0.5, 'R_Einst_max':3.0}
        type: dictionary

    -- kwargs_deflector: a dictionary containing all the foreground deflector parameters,
                     should have the following params:
                     {'min_z':0.2, 'max_z':1.9, 'min_shear':0.001, 'max_shear':0.02,
                     'min_PA_shear':0,'max_PA_shear':180.0}
        type: dictionary

    -- cosmo: the cosmology defined
        type:  astropy.cosmology

    -- constant:    a dicitionary containing the constants, Universal Gravitational constant
                    and speed of light
                    should have the parameters:
                    {'G':4.2994e-9, 'light_speed':299792.458}
        type: dictionary

    -- parallel: will be set to True only if you are running the code in parallel
                default is False

    -- save_source_file: a flag for generating the intermediate file with source info
               set to True if you want all source info file.

    -- chunk_offset: the offset for source id in case of multi-processing
              must be 0 if you are using the code in series.

    Returns:
    -- An output catalog (csv file) of the sorted galaxies that are potential deflectors
            having the same parameters as in the input catalog.
    """

    # Fix a seed
    myseed = 23524  # num*10+23424

    # open a log file to write
    srcprop.setlogfile("log.txt")

    foreground_redshift, foreground_ellipticity = (
        foreground_table["redshift"],
        foreground_table["ellipticity"],
    )
    foreground_gmag, foreground_rmag, foreground_imag = (
        foreground_table["gmag"],
        foreground_table["rmag"],
        foreground_table["imag"],
    )
    foreground_err_g, foreground_err_r, foreground_err_i = (
        foreground_table["err_g"],
        foreground_table["err_r"],
        foreground_table["err_i"],
    )
    # find the axis ratio of the foreground
    foreground_axisratio = 1 - foreground_ellipticity

    lsst_mags = np.array([foreground_gmag, foreground_rmag, foreground_imag]).T
    lsst_errs = np.array([foreground_err_g, foreground_err_r, foreground_err_i]).T

    # Get the velocity dispersion of the lensing galaxy using the light2mass model
    foreground_velocity_dispersion = get_velocity_dispersion(
        deflector_type="elliptical",
        lsst_mags=lsst_mags,
        lsst_errs=lsst_errs,
        redshift=foreground_redshift,
        cosmo=cosmo,
        bands=["g", "r", "i"],
        c1=0.01011,
        c2=0.01920,
        c3=0.05162,
        c4=-0.00032,
        c5=0.06555,
        c6=-0.02949,
        c7=0.00003,
        c8=0.04040,
        c9=-0.00892,
        c10=-0.03068,
        c11=-0.21527,
        c12=0.09394,
        scaling_relation="spectroscopic",
    )

    foreground_velocity_dispersion = unumpy.nominal_values(
        foreground_velocity_dispersion
    )

    # define the blank lists
    sorted_indices, sorted_velocity_dispersion, sorted_Einstein_radius = [], [], []
    sorted_source_mag, sorted_source_redshift = [], []
    count_source = 0

    ## Extract the lens Id, lens velocity dispersion, Einstein radius, source magnitude, source redshift for the
    ## potential deflectors
    for ii in range(len(foreground_table)):
        srcprop.fp1.write("LENSES: %d: \n" % (count_source))

        ## Choose lenses within the defined magnitude and redshift limits
        if (
            foreground_gmag[ii] > 0
            and foreground_rmag[ii] > 0
            and foreground_redshift[ii] > kwargs_deflector.get("min_z")
            and foreground_redshift[ii] < kwargs_deflector.get("max_z")
        ):

            ## Use the srcprop function to calculate the number of background sources (gal/qso) for each deflector
            ## and extract the source magnitude, redshift and the foreground velocity dispersion
            if background_source == "galaxy":
                (
                    background_source_counts,
                    source_magnitude_array,
                    source_redshift_array,
                    randoms,
                ) = srcprop.Nsrc_gal(
                    foreground_gmag[ii],
                    foreground_rmag[ii],
                    foreground_imag[ii],
                    foreground_redshift[ii],
                    foreground_axisratio[ii],
                    foreground_velocity_dispersion[ii],
                    myseed,
                    kwargs_source,
                    cosmo,
                    constants,
                )

            elif background_source == "quasar":
                (
                    background_source_counts,
                    source_magnitude_array,
                    source_redshift_array,
                    randoms,
                ) = srcprop.Nsrc_qso(
                    foreground_gmag[ii],
                    foreground_rmag[ii],
                    foreground_imag[ii],
                    foreground_redshift[ii],
                    foreground_axisratio[ii],
                    foreground_velocity_dispersion[ii],
                    myseed,
                    kwargs_source,
                    cosmo,
                    constants,
                )

            ## keep lenses with atleat one background source and >0 velocity dispersion
            if background_source_counts > 0 and foreground_velocity_dispersion[ii] > 0:
                for kk in range(background_source_counts):
                    """if(foreground_redshift[ii] > 1.2*source_redshift_array[kk]):

                    continue
                    """
                    # get the Einstein radius of the foreground deflector
                    foreground_Einstein_radius = slp.getreinst(
                        foreground_redshift[ii],
                        source_redshift_array[kk],
                        foreground_velocity_dispersion[ii],
                        cosmo,
                        constants,
                    )

                    ## Keep lenses with Einstein radius within the minimum and maximum Einstein radius defined in the kwargs
                    ## and write the output lens id, velocity dispersion, einstein radius, source mag, and
                    ## source redshift to all_sources.csv
                    if foreground_Einstein_radius * 206264.8 >= kwargs_source.get(
                        "R_Einst_min"
                    ) and foreground_Einstein_radius * 206264.8 <= kwargs_source.get(
                        "R_Einst_max"
                    ):
                        sorted_indices.append(ii + chunk_offset)
                        sorted_velocity_dispersion.append(
                            foreground_velocity_dispersion[ii]
                        )
                        sorted_Einstein_radius.append(
                            foreground_Einstein_radius * 206264.8
                        )
                        sorted_source_mag.append(source_magnitude_array[kk])
                        sorted_source_redshift.append(source_redshift_array[kk])
                        count_source += 1

    if parallel:
        print("sorted_indices ", sorted_indices)
        return (
            sorted_indices,
            sorted_velocity_dispersion,
            sorted_Einstein_radius,
            sorted_source_mag,
            sorted_source_redshift,
            count_source,
        )

    else:
        sorted_table = Table(
            [
                sorted_indices,
                sorted_velocity_dispersion,
                sorted_Einstein_radius,
                sorted_source_mag,
                sorted_source_redshift,
            ],
            names=(
                "Id",
                "Velocity dispersion",
                "Einstein radius",
                "Source mag",
                "Source z",
            ),
        )

        if save_source_file:
            sorted_table.write("all_sources.csv", format="csv", overwrite=True)
            print("Processing complete. Results saved to 'all_sources.csv'.")

        # Select rows with only unique foreground ids
        id_unique, ind_uniq = np.unique(sorted_indices, return_index=True)
        uniq_velocity_dispersion = np.array(sorted_velocity_dispersion)[ind_uniq]

        print("Found ", len(id_unique), "unique lensed ", background_source)

        # Extract only unique galaxies from the sorted galaxies
        foreground_sorted_table = foreground_table[id_unique]
        foreground_sorted_table["velocity dispersion"] = uniq_velocity_dispersion
        foreground_sorted_table["deflector Id"] = np.array(sorted_indices)[ind_uniq]

        # Save the final sorted lenses/ deflectors
        foreground_sorted_table.write(
            "sorted_galaxies.csv", format="csv", overwrite=True
        )

        return None


def find_potential_lenses_parallel(
    comm,
    rank,
    size,
    foreground_catalog,
    background_source,
    kwargs_deflector={
        "min_z": 0.2,
        "max_z": 1.9,
        "min_shear": 0.001,
        "max_shear": 0.02,
        "min_PA_shear": 0.0,
        "max_PA_shear": 180.0,
    },
    kwargs_source={
        "type": "galaxy",
        "min_z": 0.2,
        "max_z": 5.0,
        "min_mag": 25.0,
        "max_mag": 28.0,
        "boost_csect": 3,
        "ell_min": 0.1,
        "ell_max": 0.8,
    },
    cosmo=FlatLambdaCDM(H0=72, Om0=0.26),
    constants={"G": 4.2994e-9, "light_speed": 299792.458},
    save_source_file=False,
):

    # Scatter the data to all processes
    foreground_table_chunks = np.array_split(foreground_catalog, size)
    local_chunk = comm.scatter(foreground_table_chunks, root=0)

    # Calculate the offset for this chunk
    chunk_offset = sum(len(chunk) for chunk in foreground_table_chunks[:rank])

    # Each process works on its chunk
    local_results = find_potential_lenses(
        local_chunk,
        background_source,
        kwargs_deflector,
        kwargs_source,
        cosmo,
        constants,
        parallel="True",
        save_source_file=save_source_file,
        chunk_offset=chunk_offset,
    )

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Combine the results from all processes
        sorted_indices, sorted_velocity_dispersion, sorted_Einstein_radius = [], [], []
        sorted_source_mag, sorted_source_redshift = [], []

        for result in all_results:
            sorted_indices.extend(result[0])
            sorted_velocity_dispersion.extend(result[1])
            sorted_Einstein_radius.extend(result[2])
            sorted_source_mag.extend(result[3])
            sorted_source_redshift.extend(result[4])

        sorted_table = Table(
            [
                sorted_indices,
                sorted_velocity_dispersion,
                sorted_Einstein_radius,
                sorted_source_mag,
                sorted_source_redshift,
            ],
            names=(
                "Id",
                "Velocity dispersion",
                "Einstein radius",
                "Source mag",
                "Source z",
            ),
        )

        # Save intermediate source files, if needed
        if save_source_file:
            sorted_table.write("all_sources.csv", format="csv", overwrite=True)
            print("Processing complete. Results saved to 'all_sources.csv'.")

        # Select rows with only unique foreground ids
        id_unique, ind_uniq = np.unique(sorted_indices, return_index=True)
        uniq_velocity_dispersion = np.array(sorted_velocity_dispersion)[ind_uniq]

        print("Found ", len(id_unique), "unique lensed ", background_source)

        # Extract only unique galaxies from the sorted galaxies
        foreground_sorted_table = foreground_catalog[id_unique]
        foreground_sorted_table["velocity dispersion"] = uniq_velocity_dispersion

        foreground_sorted_table.write(
            "sorted_galaxies.csv", format="csv", overwrite=True
        )

    return None
