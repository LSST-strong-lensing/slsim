def galaxy_cut(galaxy_list, z_min=0, z_max=5, mag_g_max=30):
    """

    :param galaxy_list: galaxies prior to selection criteria
    :param z_min: minimum redshift of selected sample
    :param z_max: maximum redshift of selected sample
    :param mag_g_max: maximum g-band magnitude of galaxies
    :return: subset of galaxies matching the selection criteria
    """

    bool_cut = (galaxy_list['z'] > z_min) & \
               (galaxy_list['z'] < z_max) & \
               (galaxy_list['mag_g'] < mag_g_max)
    galaxy_list_cut = galaxy_list[bool_cut]
    return galaxy_list_cut
