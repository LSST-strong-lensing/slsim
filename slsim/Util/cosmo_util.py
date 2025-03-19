from astropy.cosmology import FlatLambdaCDM


def z_scale_factor(z_old, z_new, cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
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
    # Define default cosmology if provided as None
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Calculate angular diameter distance scaling factor
    return cosmo.angular_diameter_distance(z_old) / cosmo.angular_diameter_distance(
        z_new
    )
