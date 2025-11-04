def object_cut(
    galaxy_list,
    z_min=0,
    z_max=5,
    band=None,
    band_max=40,
    list_type="astropy_table",
    object_type="extended",
):
    """Selects a subset of a given galaxy list satisfying given criteria.

    :param galaxy_list: galaxies prior to selection criteria
    :param z_min: minimum redshift of selected sample
    :param z_max: maximum redshift of selected sample
    :param band: imaging band
    :param band_max: maximum magnitude of galaxies in band
    :param list_type: format of the source catalog file. Currently, it
        supports a single astropy table or a list of astropy tables.
    :param object_type: string to specify whether catalog contains an
        extended object or point object. This is necessary because point
        and extended object have different name for the magnitude.
    :return: subset of galaxies matching the selection criteria
    """
    if object_type == "extended":
        mag_string = "mag_"
    elif object_type == "point":
        mag_string = "ps_mag_"
    else:
        raise ValueError("given object type %s is not supported." % object_type)
    if list_type == "astropy_table":
        if band is None:
            bool_cut = (galaxy_list["z"] > z_min) & (galaxy_list["z"] < z_max)
        else:
            # TODO: What if you wanted to work with multiple bands?
            bool_cut = (
                (galaxy_list["z"] > z_min)
                & (galaxy_list["z"] < z_max)
                & (galaxy_list[mag_string + band] < band_max)
            )
        galaxy_list_cut = galaxy_list[bool_cut]
    else:
        galaxy_list_cut = []
        for table in galaxy_list:
            if band is None:
                bool_cut = (table["z"] > z_min) & (table["z"] < z_max)
            else:
                bool_cut = (
                    (table["z"] > z_min)
                    & (table["z"] < z_max)
                    & (table[mag_string + band] < band_max)
                )

            # Check if any rows satisfy the cut
            if any(bool_cut):
                cut_table = table[bool_cut]
                galaxy_list_cut.append(cut_table)

    return galaxy_list_cut
