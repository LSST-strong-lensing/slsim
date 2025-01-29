import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
pandas.set_option('display.max_rows', 1000)
from lsst.rsp import get_tap_service

service = get_tap_service("tap")
assert service is not None

'''
This script can be used to extract the DP0.2 Object catalogs within a defined region.
It should be run on Rubin Science Platform (RSP) only, since the adql queries 
to extract the DP0.2 catalogs is supported in RSP only.
'''

# Define the subquery function
def fetch_adql_results(service, ra_min, ra_max, dec_min, dec_max):
    '''
    Extracts galaxies from the DP0.2 Object catalog that match the truth catalog
    within a specified region of the sky, defined by right ascension (RA) and
    declination (Dec) ranges.

    Parameters:
    -----------
    service : str
        The TAP (Table Access Protocol) service endpoint used for querying 
        the DP0.2 Object catalog.
    
    ra_min : float
        The minimum right ascension (RA) of the required region, in degrees.
    
    ra_max : float
        The maximum right ascension (RA) of the required region, in degrees.
    
    dec_min : float
        The minimum declination (Dec) of the required region, in degrees.
    
    dec_max : float
        The maximum declination (Dec) of the required region, in degrees.

    Returns:
    --------
    results : pandas.dataframe
        A pandas daataframe containing the results of the ADQL query, including
        the extracted galaxy data within the specified sky region.

    '''

    query = f"""
    SELECT mt.id_truth_type AS mt_id_truth_type,
           mt.match_objectId AS mt_match_objectId,
           ts.ra AS ts_ra,
           ts.dec AS ts_dec,
           ts.truth_type AS ts_truth_type,
           ts.mag_r AS ts_mag_r,
           ts.is_pointsource AS ts_is_pointsource,
           ts.redshift AS ts_redshift,
           
           ts.flux_u AS ts_flux_u,
           ts.flux_g AS ts_flux_g,
           ts.flux_r AS ts_flux_r,
           ts.flux_i AS ts_flux_i,
           ts.flux_z AS ts_flux_z,
           ts.flux_y AS ts_flux_y,
           
           obj.coord_ra AS obj_coord_ra,
           obj.coord_dec AS obj_coord_dec,
           obj.refExtendedness AS obj_refExtendedness,
           obj.refBand AS obj_refband,

           obj.u_cModel_flag AS obj_u_cModel_flag,
           obj.u_psfFlux_flag AS obj_u_psfFlux_flag,
           obj.u_cModelFlux AS obj_u_cModelFlux,
           obj.u_cModelFluxErr AS obj_u_cModelFluxerr,
           obj.u_psfFlux AS obj_u_psfFlux,
           obj.u_psfFluxErr AS obj_u_psfFluxerr,
           obj.u_bdFluxB AS obj_u_bdFluxB,
           obj.u_bdFluxBErr AS obj_u_bdFluxBerr,
           obj.u_bdFluxD AS obj_u_bdFluxD,
           obj.u_bdFluxDErr AS obj_u_bdFluxDerr,

           obj.g_cModel_flag AS obj_g_cModel_flag,
           obj.g_psfFlux_flag AS obj_g_psfFlux_flag,
           obj.g_cModelFlux AS obj_g_cModelFlux,
           obj.g_cModelFluxErr AS obj_g_cModelFluxerr,
           obj.g_psfFlux AS obj_g_psfFlux,
           obj.g_psfFluxErr AS obj_g_psfFluxerr,
           obj.g_bdFluxB AS obj_g_bdFluxB,
           obj.g_bdFluxBErr AS obj_g_bdFluxBerr,
           obj.g_bdFluxD AS obj_g_bdFluxD,
           obj.g_bdFluxDErr AS obj_g_bdFluxDerr,

           obj.r_cModel_flag AS obj_r_cModel_flag,
           obj.r_psfFlux_flag AS obj_r_psfFlux_flag,
           obj.r_cModelFlux AS obj_r_cModelFlux,
           obj.r_cModelFluxErr AS obj_r_cModelFluxerr,
           obj.r_psfFlux AS obj_r_psfFlux,
           obj.r_psfFluxErr AS obj_r_psfFluxerr,
           obj.r_bdFluxB AS obj_r_bdFluxB,
           obj.r_bdFluxBErr AS obj_r_bdFluxBerr,
           obj.r_bdFluxD AS obj_r_bdFluxD,
           obj.r_bdFluxDErr AS obj_r_bdFluxDerr,

           obj.i_cModel_flag AS obj_i_cModel_flag,
           obj.i_psfFlux_flag AS obj_i_psfFlux_flag,
           obj.i_cModelFlux AS obj_i_cModelFlux,
           obj.i_cModelFluxErr AS obj_i_cModelFluxerr,
           obj.i_psfFlux AS obj_i_psfFlux,
           obj.i_psfFluxErr AS obj_i_psfFluxerr,
           obj.i_bdFluxB AS obj_i_bdFluxB,
           obj.i_bdFluxBErr AS obj_i_bdFluxBerr,
           obj.i_bdFluxD AS obj_i_bdFluxD,
           obj.i_bdFluxDErr AS obj_i_bdFluxDerr,

           obj.z_cModel_flag AS obj_z_cModel_flag,
           obj.z_psfFlux_flag AS obj_z_psfFlux_flag,
           obj.z_cModelFlux AS obj_z_cModelFlux,
           obj.z_cModelFluxErr AS obj_z_cModelFluxerr,
           obj.z_psfFlux AS obj_z_psfFlux,
           obj.z_psfFluxErr AS obj_z_psfFluxerr,
           obj.z_bdFluxB AS obj_z_bdFluxB,
           obj.z_bdFluxBErr AS obj_z_bdFluxBerr,
           obj.z_bdFluxD AS obj_z_bdFluxD,
           obj.z_bdFluxDErr AS obj_z_bdFluxDerr,

           obj.y_cModel_flag AS obj_y_cModel_flag,
           obj.y_psfFlux_flag AS obj_y_psfFlux_flag,
           obj.y_cModelFlux AS obj_y_cModelFlux,
           obj.y_cModelFluxErr AS obj_y_cModelFluxerr,
           obj.y_psfFlux AS obj_y_psfFlux,
           obj.y_psfFluxErr AS obj_y_psfFluxerr,
           obj.y_bdFluxB AS obj_y_bdFluxB,
           obj.y_bdFluxBErr AS obj_y_bdFluxBerr,
           obj.y_bdFluxD AS obj_y_bdFluxD,
           obj.y_bdFluxDErr AS obj_y_bdFluxDerr,

           obj.shape_xx AS obj_shape_xx,
           obj.shape_xy AS obj_shape_xy,
           obj.shape_yy AS obj_shape_yy,
           
           obj.u_bdReB AS obj_u_bdReB,
           obj.g_bdReB AS obj_g_bdReB,
           obj.r_bdReB AS obj_r_bdReB,
           obj.i_bdReB AS obj_i_bdReB,
           obj.z_bdReB AS obj_z_bdReB,
           obj.y_bdReB AS obj_y_bdReB,

           obj.u_bdReD AS obj_u_bdReD,
           obj.g_bdReD AS obj_g_bdReD,
           obj.r_bdReD AS obj_r_bdReD,
           obj.i_bdReD AS obj_i_bdReD,
           obj.z_bdReD AS obj_z_bdReD,
           obj.y_bdReD AS obj_y_bdReD
           
    FROM dp02_dc2_catalogs.MatchesTruth AS mt
    JOIN dp02_dc2_catalogs.TruthSummary AS ts ON mt.id_truth_type = ts.id_truth_type
    JOIN dp02_dc2_catalogs.Object AS obj ON mt.match_objectId = obj.objectId
    WHERE CONTAINS(POINT('ICRS', obj.coord_ra, obj.coord_dec), 
                   POLYGON('ICRS', {ra_min}, {dec_min}, 
                                  {ra_min}, {dec_max}, 
                                  {ra_max}, {dec_max}, 
                                  {ra_max}, {dec_min})) = 1
    AND ts.truth_type = 1
    AND obj.refExtendedness = 1
    AND obj.u_cModelFlux > 0
    AND obj.g_cModelFlux > 0
    AND obj.r_cModelFlux > 0
    AND obj.i_cModelFlux > 0
    AND obj.z_cModelFlux > 0
    AND obj.y_cModelFlux > 0
    AND obj.u_cModel_flag = 0
    AND obj.g_cModel_flag = 0
    AND obj.r_cModel_flag = 0
    AND obj.i_cModel_flag = 0
    AND obj.z_cModel_flag = 0
    AND obj.y_cModel_flag = 0
    AND obj.detect_isPrimary = 1
    """

    job = service.submit_job(query)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    print('Job phase is', job.phase)

    # Fetch and return results if the job is completed
    if job.phase == 'COMPLETED':
        results = job.fetch_result().to_table().to_pandas()
        print(f"Number of results: {len(results)}")
        return results
    else:
        print("Job failed with error phase.")
        return None
    

# Provide the ra and dec range within which the catalog has to be extracted
ra_min,dec_min,ra_max,dec_max = 71.875, -28.125, 75.0, -25.0

query_results = fetch_adql_results(service, ra_min, ra_max, dec_min, dec_max)

object_table = query_results                                                       

# Write the extracted catalog to a csv file.
object_table.to_csv('DP0p2_final_%f_%f_%f_%f.csv'%(ra_min,ra_max,dec_min,dec_max))