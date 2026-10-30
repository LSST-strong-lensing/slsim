import numpy as np
from astropy import units as u
from astropy.table import Table

from skypy.galaxies.morphology import beta_ellipticity

from slsim.Util.param_util import flux_to_ab_magnitude, galaxy_size_redshift_evolution


def prepare_source_catalog(
    catalog, column_mapping=None, flux_columns=None, flux_unit=None, fill_missing=None
):
    """Converts an external source catalog into the conventions of the SLSim
    source populations (e.g. Galaxies() with catalog_type="external").

    SLSim requires the redshift "z", AB magnitudes "mag_<band>" for the
    bands used in the selection and, for Sersic profiles, a size
    ("angular_size" [arcsec] or "physical_size" [kpc]) and a shape ("e1"
    and "e2", or "ellipticity").

    :param catalog: input catalog
    :type catalog: astropy Table, pandas DataFrame or dict of arrays
    :param column_mapping: renaming of columns into SLSim names, e.g.
        {"Z": "z", "MAG_I": "mag_i"}
    :param flux_columns: flux columns to convert into AB magnitudes, as
        {column: band}, e.g. {"FLUX_I": "i"} adds "mag_i". Non-positive
        fluxes get an infinite magnitude.
    :param flux_unit: astropy unit of the flux columns without an
        attached unit
    :param fill_missing: quantities not in the catalog to draw from a
        model. Supported are "physical_size" (median size-redshift
        relation of Shibuya et al. 2015, derived from rest-frame
        UV/optical sizes) and "ellipticity" (beta distribution of
        Kacprzak et al. 2019 with the SLSim SkyPy parameters for star-
        forming galaxies).
    :type fill_missing: list of str
    :return: catalog in SLSim conventions
    :rtype: astropy Table
    """
    table = Table(catalog, copy=True)
    for old_name, new_name in (column_mapping or {}).items():
        table.rename_column(old_name, new_name)

    for col_name, band in (flux_columns or {}).items():
        flux = u.Quantity(table[col_name], flux_unit)
        mag = flux_to_ab_magnitude(flux)
        table["mag_" + band] = np.where(flux.value > 0, mag, np.inf)

    for col_name in fill_missing or []:
        if col_name == "physical_size":
            table[col_name] = galaxy_size_redshift_evolution(table["z"])
        elif col_name == "ellipticity":
            table[col_name] = beta_ellipticity(e_ratio=0.45, e_sum=3.5, size=len(table))
        else:
            raise ValueError(
                "No model to fill '%s'; only 'physical_size' and 'ellipticity' are "
                "supported. Please add it to the catalog." % col_name
            )
    return table
