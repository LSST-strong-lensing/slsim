from astropy.cosmology import FlatLambdaCDM

def z_scale_factor(z_old, z_new, cosmo=None):
    """Return multiplication factor for object/pixel size for moving its
    redshift from z_old to z_new.

    Args:
        z_old (float): The original redshift of the object.
        z_new (float): The redshift the object will be placed at.
        cosmo (~astropy.cosmology.FLRW, optional): The cosmology object. Defaults to a FlatLambdaCDM model.

    Returns:
        (float): The multiplicative pixel size.
    """
    # Define default cosmology if not provided
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    # Calculate angular diameter distance scaling factor
    return (
        cosmo.angular_diameter_distance(z_old)
        / cosmo.angular_diameter_distance(z_new)
    )
