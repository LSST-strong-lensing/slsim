import numpy as np
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import Planck13, FlatLambdaCDM
from sklearn.neighbors import NearestNeighbors

def merge_catalogs(large_cat, small_cat,
                   lc_mag_col, sc_mag_col,
                   tolerance=1.0,
                   lc_ra_col='RA', lc_dec_col='DEC',
                   sc_ra_col='RA', sc_dec_col='DEC'):
    # SkyCoord objects
    Lcoords = SkyCoord(ra=large_cat[lc_ra_col], dec=large_cat[lc_dec_col])
    Scoords = SkyCoord(ra=small_cat[sc_ra_col], dec=small_cat[sc_dec_col])
    # match
    idx, d2d, _ = Scoords.match_to_catalog_sky(Lcoords)
    d2d_arcsec = d2d.to(u.arcsec)
    # mask by tolerance
    mask = d2d_arcsec < tolerance * u.arcsec
    # compute MAG_DIFF on matched
    small_cat['MAG_DIFF'] = np.nan
    small_cat['MAG_DIFF'][mask] = (small_cat[sc_mag_col][mask]
                                   - large_cat[lc_mag_col][idx][mask])
    # stack matched rows
    return hstack([large_cat[idx[mask]], small_cat[mask]])


def match_simulated_to_real(
    sim_table,
    cosmos_catalog,
    cosmo,
    tolerance=None,
    mag_col_sim='mag_i',
    size_col_sim='physical_size',
    mag_col_cat='MAGabs',
    size_col_cat='RHALFreal',
    id_col_sim=None,
    id_col_cat=None,
    n_neighbors=1,
):
    """
    Match simulated SkyPy sources to real COSMOS catalog sources using nearest-neighbor
    matching in magnitude-size feature space, with an optional distance tolerance filter.

    Returns an Astropy table containing all columns from sim_table and matched
    columns from the COSMOS catalog (prefixed 'COSMOS_'), along with 'MatchedIndex'
    and 'Distance'.
    """
    # Load catalog
    if isinstance(cosmos_catalog, str):
        cat_table = Table.read(cosmos_catalog)
    elif isinstance(cosmos_catalog, Table):
        cat_table = cosmos_catalog
    else:
        raise ValueError("cosmos_catalog must be a filepath or an astropy.table.Table")

    # Helper to extract raw values
    def _values(arr):
        try:
            return np.array([val.value if hasattr(val, 'unit') else val for val in arr])
        except Exception:
            return np.array(arr)

    # Compute simulated absolute magnitude if needed
    if mag_col_sim == 'mag_i':
        if 'z' not in sim_table.colnames:
            raise KeyError("'z' column required for computing absolute magnitude")
        zvals = sim_table['z']
        ang = cosmo.angular_diameter_distance(zvals)
        sim_mag = sim_table['mag_i'] + 5 - 5 * np.log10(ang.value * 1e6)
    else:
        sim_mag = _values(sim_table[mag_col_sim])

    # Extract sizes
    sim_size = _values(sim_table[size_col_sim])
    cat_mag = _values(cat_table[mag_col_cat])
    cat_size = _values(cat_table[size_col_cat])

    # Normalize features
    def _norm(x):
        rng = np.max(x) - np.min(x)
        return (x - np.min(x)) / rng if rng != 0 else np.zeros_like(x)

    sim_feats = np.column_stack((_norm(sim_mag), _norm(sim_size)))
    cat_feats = np.column_stack((_norm(cat_mag), _norm(cat_size)))

    # Nearest-neighbor search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(cat_feats)
    distances, indices = nbrs.kneighbors(sim_feats)
    distances = distances.flatten()
    indices = indices.flatten()

    # Prepare simulated output with full-length arrays
    sim_out = sim_table.copy()
    sim_out['MatchedIndex'] = indices
    sim_out['Distance'] = distances
    if id_col_sim and id_col_sim in sim_out.colnames:
        sim_out.rename_column(id_col_sim, id_col_sim)

    # Apply tolerance filter to both sim and cat entries
    if tolerance is not None:
        mask = distances <= tolerance
    else:
        mask = np.ones_like(distances, dtype=bool)

    sim_masked = sim_out[mask]
    cat_matched = cat_table[indices[mask]]

    # Prefix matched catalog columns
    cat_prefixed = cat_matched.copy()
    for col in cat_prefixed.colnames:
        cat_prefixed.rename_column(col, f'COSMOS_{col}')

    # Combine and return
    result = hstack([sim_masked, cat_prefixed])
    return result



def z_scale_factor(z_old, z_new, cosmo):
    """
    :param z_old: The original redshift.
    :type z_old: float

    :param z_new: The redshift where the object will be placed.
    :type z_new: float

    :param cosmo: The cosmology object. Defaults to a FlatLambdaCDM model if None.
    :type cosmo: astropy.cosmology.FLRW, optional

    :return: The multiplicative pixel size scaling factor.
    :rtype: float
    """
    # Calculate angular diameter distance scaling factor
    return cosmo.angular_diameter_distance(z_old) / cosmo.angular_diameter_distance(
        z_new
    )

def get_cosmos_catalog():
    # Adjust path to the COSMOS catalog
    return Table.read("/path/to/cosmos_catalog.fits")

def load_cosmos_image(filename, hdu_index):
    with fits.open(filename) as hdulist:
        return hdulist[hdu_index].data

def flux_weighted_center(image):
    y, x = np.indices(image.shape)
    total = image.sum()
    if total <= 0:
        return image.shape[1] / 2, image.shape[0] / 2
    x_center = (x * image).sum() / total
    y_center = (y * image).sum() / total
    return x_center, y_center

