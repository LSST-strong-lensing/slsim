import argparse
import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
import lsst.daf.butler as dafButler
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.lens_pop import LensPop
import slsim.Pipelines as pipelines
import slsim.Sources as sources
import slsim.Deflectors as deflectors

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    # Create a simple fallback if tqdm is not available
    HAS_TQDM = False

    def tqdm(iterable, **kwargs):
        """Simple fallback for tqdm progress bar when the library is not available.

        :param iterable: Iterable to iterate over
        :type iterable: iterable
        :param kwargs: Keyword arguments that would be passed to tqdm (ignored in fallback)
        :return: Unchanged input iterable
        :rtype: iterable
        """
        return iterable




def generate_master_galaxy_list(
    ra_min, ra_max, dec_min, dec_max, n_galaxies=1000, sky_area_value=0.15
):
    """Generate a master list of galaxies (source and lens types) using slsim.

    :param ra_min: Minimum right ascension in degrees
    :type ra_min: float
    :param ra_max: Maximum right ascension in degrees
    :type ra_max: float
    :param dec_min: Minimum declination in degrees
    :type dec_min: float
    :param dec_max: Maximum declination in degrees
    :type dec_max: float
    :param n_galaxies: Number of galaxies to generate
    :type n_galaxies: int
    :param sky_area_value: Sky area in square degrees for galaxy simulation
    :type sky_area_value: float
    :return: Galaxy catalog
    :rtype: `astropy.table.Table`
    """
    # Define cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Define sky area for galaxy simulation and lens population
    sky_area = Quantity(value=sky_area_value, unit="deg2")
    # For this implementation, we use the same area for both, but they could be different
    full_sky_area = Quantity(value=sky_area_value, unit="deg2")

    # Define cuts for galaxy selection
    kwargs_deflector_cut = {"band": "i", "band_max": 27, "z_min": 0.1, "z_max": 2}
    kwargs_source_cut = {"band": "i", "band_max": 27, "z_min": 0.1, "z_max": 5}

    # Initialize galaxy simulation pipeline
    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=None, sky_area=sky_area, filters=None, cosmo=cosmo
    )

    # Initialize lens population
    lens_galaxies = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light=None,
        cosmo=cosmo,
        sky_area=sky_area,
    )

    # Initialize source population
    source_galaxies = sources.Galaxies(
        galaxy_list=galaxy_simulation_pipeline.blue_galaxies,
        kwargs_cut=kwargs_source_cut,
        cosmo=cosmo,
        sky_area=sky_area,
        catalog_type="skypy",
    )

    # Create lens population
    lenspop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_galaxies,
        cosmo=cosmo,
        sky_area=full_sky_area,
    )

    # Get lens and source galaxies
    lens = lenspop._lens_galaxies._galaxy_select
    source = lenspop._sources._galaxy_select

    # Check if we have enough galaxies for the requested catalog size
    if len(lens) < n_galaxies or len(source) < n_galaxies:
        raise ValueError(
            f"Not enough galaxies generated. Requested {n_galaxies} of each type, but "
            f"only got {len(lens)} lens galaxies and {len(source)} source galaxies. "
            f"Try increasing sky_area_value or decreasing n_galaxies."
        )

    # Process lens galaxies
    lens_ell_mask = lens["n_sersic"] < -0.999
    n_ell_lens = np.sum(lens_ell_mask)
    
    if n_ell_lens > 0:
        phi = np.random.uniform(0, np.pi, size=n_ell_lens)
        e = lens["ellipticity"][lens_ell_mask].data
        ep = (1 - np.sqrt(1 - e**2)) / e
        e1 = ep * np.cos(2 * phi)
        e2 = ep * np.sin(2 * phi)
        lens["e1_light"][lens_ell_mask] = e1
        lens["e2_light"][lens_ell_mask] = e2
        lens["n_sersic"][lens_ell_mask] = 4

    # Process source galaxies
    source_ell_mask = source["n_sersic"] < -0.999
    n_ell_source = np.sum(source_ell_mask)

    if n_ell_source > 0:
        phi = np.random.uniform(0, np.pi, size=n_ell_source)
        e = source["ellipticity"][source_ell_mask].data
        ep = (1 - np.sqrt(1 - e**2)) / e
        e1 = ep * np.cos(2 * phi)
        e2 = ep * np.sin(2 * phi)
        source["e1"][source_ell_mask] = e1
        source["e2"][source_ell_mask] = e2
        source["n_sersic"][source_ell_mask] = 1


    # Generate random positions within the specified RA/Dec box
    ra = np.random.uniform(ra_min, ra_max, size=n_galaxies)
    dec = np.random.uniform(dec_min, dec_max, size=n_galaxies)

    # Initialize ID, RA, Dec fields
    source["ra"] = np.nan
    source["dec"] = np.nan
    source["id"] = -1
    source["type"] = "source"
    source["source_type"] = "extended"
    source["light_profile"] = "single_sersic"
    lens["ra"] = np.nan
    lens["dec"] = np.nan
    lens["id"] = -1
    lens["type"] = "lens"
    lens["deflector_type"] = "EPL"

    # Randomly select galaxies and assign positions
    lens_index = np.random.choice(range(len(lens)), n_galaxies, replace=False)
    source_index = np.random.choice(range(len(source)), n_galaxies, replace=False)

    source["id"][source_index] = np.arange(0, n_galaxies).astype(int)
    lens["id"][lens_index] = np.arange(0, n_galaxies).astype(int)
    source["ra"][source_index] = ra
    source["dec"][source_index] = dec
    lens["ra"][lens_index] = ra
    lens["dec"][lens_index] = dec

    # Combine sources and lenses and expand coefficient array
    gals = vstack([source[source_index], lens[lens_index]])

    # Handle coefficient array expansion
    if "coeff" in gals.colnames:
        if len(gals["coeff"].shape) > 1 and gals["coeff"].shape[1] >= 5:
            gals["coeff0"] = gals["coeff"][:, 0]
            gals["coeff1"] = gals["coeff"][:, 1]
            gals["coeff2"] = gals["coeff"][:, 2]
            gals["coeff3"] = gals["coeff"][:, 3]
            gals["coeff4"] = gals["coeff"][:, 4]
            del gals["coeff"]
        else:
            # Handle the case when coeffs don't have expected shape
            for i in range(5):
                if f"coeff{i}" not in gals.colnames:
                    gals[f"coeff{i}"] = np.zeros(len(gals))
    else:
        # If coeffs don't exist, create empty ones
        for i in range(5):
            gals[f"coeff{i}"] = np.zeros(len(gals))

    return gals


def get_calexps_in_region(
    butler,
    collection,
    time_range,
    ra_range,
    dec_range,
    instrument="LSSTComCam",
    filters=["u", "g", "r", "i", "z", "y"],
    max_calexps=10,
):
    """Get calexp objects from the butler within given time and spatial constraints.

    :param butler: The butler instance
    :type butler: `lsst.daf.butler.Butler`
    :param collection: The collection to query
    :type collection: str
    :param time_range: (start_time, end_time) as astropy.time.Time objects
    :type time_range: tuple
    :param ra_range: (ra_min, ra_max) in degrees
    :type ra_range: tuple
    :param dec_range: (dec_min, dec_max) in degrees
    :type dec_range: tuple
    :param instrument: The instrument name to query
    :type instrument: str
    :param filters: List of filters to include
    :type filters: list
    :param max_calexps: Maximum number of calexps to process
    :type max_calexps: int
    :return: Tuple containing visit_list, visit_ras, visit_decs, visit_times
    :rtype: tuple
    """
    # Query for available visit IDs with the filter constraint
    filter_conditions = []
    for f in filters:
        filter_conditions.append(f"band='{f}'")

    filter_clause = " OR ".join(filter_conditions)
    where_clause = f"instrument='{instrument}' AND ({filter_clause})"

    print(f"Querying visits with where clause: {where_clause}")
    try:
        # First, query for visits
        visitIds = butler.registry.queryDataIds(
            ["visit"],
            datasets="visitSummary",
            collections=collection,
            where=where_clause,
        )
        print(f"Successfully queried visit IDs")
    except Exception as e:
        print(f"Error querying visits: {e}")
        return [], [], [], []

    # Get visitSummaries and filter by time and spatial overlap
    time_start = time_range[0]
    time_end = time_range[1]

    # Convert visitIds to a list for progress tracking
    visit_list = list(visitIds)
    total_visits = len(visit_list)
    print(f"Found {total_visits} visits to check")

    # Process each visit summary
    visit_ras = []
    visit_decs = []
    visit_times = []

    for visit_dataId in tqdm(visit_list, desc="Checking visitSummaries", unit="visit"):
        try:
            # Get the visitSummary
            visit_id = visit_dataId["visit"]
            visitSummary = butler.get(
                "visitSummary", visit_dataId, collections=collection
            )

            # Get the date from day_obs
            try:
                day_obs = visit_dataId["day_obs"]
                visit_str = str(visit_dataId["visit"])
                print(visit_str)
                if len(visit_str) >= 14:  # Full timestamp format
                    time_part = visit_str[-6:]  # Extract HHMMSS
                    hour = int(time_part[:2])
                    minute = int(time_part[2:4])
                    second = int(time_part[4:6])
                    # Create date string in ISO format
                    date_str = f"{day_obs // 10000:04d}-{(day_obs // 100) % 100:02d}-{day_obs % 100:02d}T{hour:02d}:{minute:02d}:{second:02d}"
                    visit_date = Time(date_str, format="isot")
                else:
                    # Just use the day with time 00:00:00
                    date_str = f"{day_obs // 10000:04d}-{(day_obs // 100) % 100:02d}-{day_obs % 100:02d}T00:00:00"
                    visit_date = Time(date_str, format="isot")

                # Check if it's in our time range
                if visit_date < time_start or visit_date > time_end:
                    continue
            except Exception:
                continue

            # Filter already checked in the query

            # We'll check each detector individually for overlap with our region

            # Check each detector in the visit for overlap with our region
            detector_ids = visitSummary["id"]

            # Initialize min/max values for this visit
            min_ra_visit = np.inf
            max_ra_visit = -np.inf
            min_dec_visit = np.inf
            max_dec_visit = -np.inf

            try:
                for i, detector_id in enumerate(detector_ids):
                    ra_corners = visitSummary["raCorners"][i]
                    dec_corners = visitSummary["decCorners"][i]
                    min_ra, max_ra = min(ra_corners), max(ra_corners)
                    min_dec, max_dec = min(dec_corners), max(dec_corners)

                    # Update visit-level min/max values
                    min_ra_visit = min(min_ra, min_ra_visit)
                    max_ra_visit = max(max_ra, max_ra_visit)
                    min_dec_visit = min(min_dec, min_dec_visit)
                    max_dec_visit = max(max_dec, max_dec_visit)
            except Exception:
                continue

            # Add this visit's data to our lists
            visit_ras.extend((min_ra_visit, max_ra_visit))
            visit_decs.extend((min_dec_visit, max_dec_visit))
            visit_times.append(visit_date)

        except Exception:
            pass

    return visit_list, visit_ras, visit_decs, visit_times


def find_galaxies_for_each_visit(
    visit_list, visit_ras, visit_decs, visit_times, galaxy_catalog, reference_time=None
):
    """Find which galaxies from the master catalog overlap with each visit's boundaries and record the visit ID and time information.

    :param visit_list: List of visit dataIds
    :type visit_list: list
    :param visit_ras: List of RA values (min/max pairs) for visits
    :type visit_ras: list
    :param visit_decs: List of Dec values (min/max pairs) for visits
    :type visit_decs: list
    :param visit_times: List of visit timestamps
    :type visit_times: list
    :param galaxy_catalog: The master galaxy catalog
    :type galaxy_catalog: `astropy.table.Table`
    :param reference_time: Reference time to compute relative times
    :type reference_time: `astropy.time.Time`
    :return: Table containing galaxies with their overlapping visits
    :rtype: `astropy.table.Table`
    """
    # Prepare to store all galaxies with their visit information
    all_overlaps = []

    # Set reference time if not provided
    if reference_time is None and visit_times:
        reference_time = min(visit_times)

    # Process each visit
    for i, (visit_dataId, visit_time) in enumerate(zip(visit_list, visit_times)):
        visit_id = visit_dataId["visit"]

        # Calculate index into the ra/dec lists, which have min/max pairs for each visit
        # Each visit has 2 entries (min/max) in the ra/dec lists
        idx = i * 2

        # Get the RA/Dec bounds for this visit
        # Make sure we don't go beyond list bounds
        if idx + 1 < len(visit_ras) and idx + 1 < len(visit_decs):
            min_ra = visit_ras[idx]
            max_ra = visit_ras[idx + 1]
            min_dec = visit_decs[idx]
            max_dec = visit_decs[idx + 1]
        else:
            continue  # Skip if indices are out of bounds

        # Find galaxies within these bounds
        mask = (
            (galaxy_catalog["ra"] >= min_ra)
            & (galaxy_catalog["ra"] <= max_ra)
            & (galaxy_catalog["dec"] >= min_dec)
            & (galaxy_catalog["dec"] <= max_dec)
        )

        visit_galaxies = galaxy_catalog[mask].copy()

        # Skip if no galaxies found
        if len(visit_galaxies) == 0:
            continue

        # Add visit information to each galaxy
        visit_galaxies["visit"] = visit_id

        # Store observation time as ISO format string
        visit_galaxies["observation_time"] = visit_time.isot

        # Add relative time if reference_time is provided
        if reference_time is not None:
            time_delta = (visit_time - reference_time).sec
            visit_galaxies["time_delta"] = time_delta

        all_overlaps.append(visit_galaxies)

    # Combine all visit-galaxy pairs
    if all_overlaps:
        return vstack(all_overlaps)
    else:
        # Return empty table with same structure
        if len(galaxy_catalog) > 0:
            empty_table = galaxy_catalog[:0].copy()
            empty_table["visit"] = []
            # Create observation_time as string column
            empty_table["observation_time"] = np.array(
                [], dtype="S26"
            )  # ISO format datetime string
            if reference_time is not None:
                empty_table["time_delta"] = []
            return empty_table
        else:
            return Table()  # Empty table


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a catalog of galaxies and find overlaps with LSST visits."
    )
    
    # Region of interest
    parser.add_argument("--ra-min", type=float, default=52, help="Minimum right ascension in degrees")
    parser.add_argument("--ra-max", type=float, default=54, help="Maximum right ascension in degrees")
    parser.add_argument("--dec-min", type=float, default=-29, help="Minimum declination in degrees")
    parser.add_argument("--dec-max", type=float, default=-27, help="Maximum declination in degrees")
    
    # Time range
    parser.add_argument(
        "--start-time", 
        type=str, 
        default="2024-11-01T00:00:00", 
        help="Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--end-time", 
        type=str, 
        default="2024-12-29T00:00:00", 
        help="End time in ISO format (YYYY-MM-DDTHH:MM:SS)"
    )
    
    # Galaxy generation parameters
    parser.add_argument(
        "--n-galaxies", 
        type=int, 
        default=10000, 
        help="Number of galaxies to generate"
    )
    parser.add_argument(
        "--sky-area", 
        type=float, 
        default=5.0, 
        help="Sky area in square degrees for galaxy simulation"
    )
    
    # Butler parameters
    parser.add_argument(
        "--repo-path", 
        type=str, 
        default="/repo/main", 
        help="Path to the butler repository"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="LSSTComCam/runs/DRP/DP1/w_2025_07/DM-48940", 
        help="Butler collection to query"
    )
    parser.add_argument(
        "--instrument", 
        type=str, 
        default="LSSTComCam", 
        help="Instrument name to query"
    )
    parser.add_argument(
        "--filters", 
        type=str, 
        nargs="+", 
        default=["r"], 
        help="List of filters to include"
    )
    
    # Output
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="galaxy_visit_overlaps.fits", 
        help="Output file name for galaxy-visit overlaps"
    )
    
    return parser.parse_args()


def main():
    """Main function to process calexps and find overlapping galaxies."""
    args = parse_args()
    
    # Define region of interest
    ra_min, ra_max = args.ra_min, args.ra_max
    dec_min, dec_max = args.dec_min, args.dec_max

    # Define time range
    start_time = Time(args.start_time, format="isot")
    end_time = Time(args.end_time, format="isot")

    # Generate master galaxy catalog with sky area parameter
    print("Generating master galaxy catalog...")
    sky_area_value = args.sky_area
    n_galaxies = args.n_galaxies
    
    master_galaxies = generate_master_galaxy_list(
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        n_galaxies=n_galaxies,
        sky_area_value=sky_area_value,
    )
    print(f"Generated {len(master_galaxies)} galaxy entries in catalog")

    # Connect to the butler repository
    print("Connecting to butler repository...")
    repo_path = args.repo_path
    butler = dafButler.Butler(repo_path)

    # Get visit information
    print("Fetching visit information in the specified region and time range...")
    collection = args.collection
    instrument = args.instrument
    filters = args.filters
    print(f"Limiting search to filters: {filters}")

    visit_list, visit_ras, visit_decs, visit_times = get_calexps_in_region(
        butler,
        collection,
        (start_time, end_time),
        (ra_min, ra_max),
        (dec_min, dec_max),
        instrument=instrument,
        filters=filters,
    )

    print(f"Found {len(visit_list)} visits with valid time and spatial information")

    if not visit_times:
        print("No visits found. Exiting.")
        return

    # Find reference time
    reference_time = min(visit_times) if visit_times else None
    if reference_time:
        print(f"Reference time set to: {reference_time}")
    else:
        print("No reference time available")
        return

    # Find galaxies overlapping with each visit and add visit information
    overlapping_galaxies = find_galaxies_for_each_visit(
        visit_list, visit_ras, visit_decs, visit_times, master_galaxies, reference_time
    )

    # Count unique galaxies and total overlaps
    unique_galaxy_count = (
        len(set(overlapping_galaxies["id"])) if len(overlapping_galaxies) > 0 else 0
    )
    total_overlaps = len(overlapping_galaxies)

    print(
        f"Found {unique_galaxy_count} unique galaxies with {total_overlaps} total overlaps across visits"
    )

    # Save results
    if total_overlaps > 0:
        # Save the results
        output_file = args.output_file
        overlapping_galaxies.write(output_file, overwrite=True)
        print(f"Saved {total_overlaps} galaxy-visit overlaps to {output_file}")
    else:
        print("No overlapping galaxies found")


if __name__ == "__main__":
    main()
