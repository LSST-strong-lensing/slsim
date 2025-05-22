import numpy as np
import pytest
import hashlib
import pickle
from pathlib import Path

from astropy.table import Table
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from slsim.Util.cosmo_util import merge_catalogs, match_simulated_to_real, flux_weighted_center
from slsim.Util import real_to_sim_matching as r2s


def make_cat(ra_list, dec_list, mag_list):
    """
    Helper: build a simple Astropy Table with RA, DEC (deg) and MAG columns.
    """
    tbl = Table()
    tbl['RA'] = ra_list * u.deg
    tbl['DEC'] = dec_list * u.deg
    tbl['MAG'] = mag_list
    return tbl


def test_merge_catalogs_basic():
    """
    Test that merge_catalogs merges matching entries and computes MAG_DIFF correctly.
    """
    large = make_cat(np.array([10, 20]), np.array([0, 0]), np.array([15.0, 16.0]))
    small = make_cat(np.array([10]), np.array([0]), np.array([14.5]))
    merged = merge_catalogs(
        large, small,
        lc_mag_col='MAG',
        sc_mag_col='MAG',
        tolerance=1.0
    )
    assert len(merged) == 1
    assert 'MAG_DIFF' in merged.colnames
    diff = merged['MAG_DIFF'][0]
    assert np.isclose(diff, 14.5 - 15.0)


def test_match_simulated_to_real_basic(tmp_path):
    """
    Test match_simulated_to_real with a minimal simulated table and real catalog.
    """
    sim = Table()
    sim['mag_i'] = [20.0, 21.0]
    sim['z'] = [0.5, 1.0]
    sim['physical_size'] = [1.0, 2.0]

    real = Table()
    real['MAGabs'] = [20.1, 20.9]
    real['RHALFreal'] = [1.0, 2.0]

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    out = match_simulated_to_real(
        sim, real,
        cosmo=cosmo,
        tolerance=None,
        n_neighbors=1,
    )

    assert len(out) == 2
    assert 'COSMOS_MAGabs' in out.colnames
    assert 'Distance' in out.colnames


def test_flux_weighted_center_zero():
    """
    Test that flux_weighted_center returns image center if all pixels are zero.
    """
    img = np.zeros((10, 20))
    x, y = flux_weighted_center(img)
    assert np.isclose(x, 20 / 2)
    assert np.isclose(y, 10 / 2)


def test_flux_weighted_center_nonzero():
    """
    Test flux_weighted_center on an image with a single bright pixel.
    """
    img = np.zeros((5, 5))
    img[2, 3] = 10
    x, y = flux_weighted_center(img)
    assert np.isclose(x, 3)
    assert np.isclose(y, 2)


def test_fingerprint_consistency(tmp_path):
    """
    Test that _fingerprint yields consistent hashes for same inputs
    and different hashes for different inputs.
    """
    val1 = r2s._fingerprint('a', 123, tmp_path)
    val2 = r2s._fingerprint('a', 123, tmp_path)
    assert val1 == val2

    val3 = r2s._fingerprint('b', 123, tmp_path)
    assert val1 != val3


def test_values_with_units():
    """
    Test that _values strips Astropy units and returns a numpy array.
    """
    arr = np.array([1, 2, 3]) * u.kpc
    out = r2s._values(arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind in ('i', 'f')

    arr2 = np.array([4, 5, 6])
    out2 = r2s._values(arr2)
    assert np.array_equal(out2, arr2)


def test_normalise():
    """
    Test that _normalise scales arrays to [0, 1], and handles constant arrays.
    """
    x = np.array([2.0, 4.0, 6.0])
    norm_x = r2s._normalise(x)
    assert np.allclose(norm_x, [0.0, 0.5, 1.0])

    y = np.array([5, 5, 5])
    norm_y = r2s._normalise(y)
    assert np.allclose(norm_y, np.zeros_like(y))


def test_build_matched_table_simple(tmp_path):
    """
    Smoke-test build_matched_table using a dummy sim_cat_path to avoid
    build_sim_catalog, and ECSV inputs for the real catalog.
    """
    from astropy.table import Table
    from astropy import units as u

    # Paths for catalogs
    cat1 = tmp_path / "c1.fits"
    cat2 = tmp_path / "c2.fits"
    morph = tmp_path / "morph.ecsv"
    sim_fits = tmp_path / "sim.fits"

    # 1) Real catalogs (cat1 & cat2) and morpho (small) for build_real_catalog
    t1 = Table({
        'IDENT':       [1],
        'MAG':         [20.0],
        'RA':          [10.0],
        'DEC':         [0.0],
        'zphot':       [0.5],
        'R_HALF':      [5.0],
        'PIXEL_SCALE': [0.05],
    })
    t2 = Table({
        'IDENT':        [1],
        'MAG_AUTO_ACS': [19.5],
    })
    morpho = Table({
        'RA':            [10.0] * u.deg,
        'DEC':           [0.0]  * u.deg,
        'MAG_AUTO_ACS':  [19.5],
        'sersicfit':     [[0, 0, 0, 0, 0, 30]],
    })

    t1.write(cat1, format='fits', overwrite=True)
    t2.write(cat2, format='fits', overwrite=True)
    morpho.write(morph, format='ascii.ecsv', overwrite=True)

    # 2) Dummy simulation catalog
    sim_tab = Table({
        'mag_i':          [20.0],
        'z':              [0.5],
        'physical_size':  [1.0],
    })
    sim_tab.write(sim_fits, format='fits', overwrite=True)

    # 3) Invoke build_matched_table with sim_cat_path
    matched, sim_out, real_out = r2s.build_matched_table(
        catalog_paths={
            'cat1': str(cat1),
            'cat2': str(cat2),
            'morpho': str(morph),
        },
        cosmo_kwargs={'H0': 70, 'Om0': 0.3},
        sky_area_deg2=0.1,
        source_params={'source_exclusion_list': []},
        sim_cat_path=str(sim_fits),
        return_tables=True,
        cache=False,
    )

    # Basic assertions
    assert isinstance(matched, Table)
    assert isinstance(sim_out, Table)
    assert isinstance(real_out, Table)
    assert 'mag_i'         in sim_out.colnames
    assert 'physical_size' in sim_out.colnames
    assert 'IDENT'         in real_out.colnames
    assert 'zphot'         in real_out.colnames
    assert 'MAG'          in real_out.colnames
    assert 'RHALF'      in real_out.colnames
    assert 'PIXEL_SCALE' in real_out.colnames
    assert any(col.startswith('COSMOS_phi_G') for col in matched.colnames)











