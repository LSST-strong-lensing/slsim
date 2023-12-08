def deflector_cut(galaxy_list, z_min=0, z_max=5, band=None, band_max=40):
    """Selects a subset of a given galaxy list satisfying given criteria.

    :param galaxy_list: galaxies prior to selection criteria
    :param z_min: minimum redshift of selected sample
    :param z_max: maximum redshift of selected sample
    :param band: imaging band
    :param band_max: maximum magnitude of galaxies in band
    :return: subset of galaxies matching the selection criteria
    """
    if band is None:
        bool_cut = (galaxy_list["z"] > z_min) & (galaxy_list["z"] < z_max)
    else:
        bool_cut = (
            (galaxy_list["z"] > z_min)
            & (galaxy_list["z"] < z_max)
            & (galaxy_list["mag_" + band] < band_max)
        )
    galaxy_list_cut = galaxy_list[bool_cut]
    return galaxy_list_cut
