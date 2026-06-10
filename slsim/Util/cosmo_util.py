import numpy as np
import scipy.interpolate as interp


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


def z_time_interp(cosmo, z_max):
    """Calculates redshift given cosmic time.

    :param cosmo: cosmology used to calculate cosmic time
    :type cosmo: astropy.cosmology object
    :param z_max: maximum redshift for interpolation
    :type z_max: float
    :return: interpolation function that returns redshift for a given
        cosmic time
    :return type: scipy.interpolate.interp1d
    """
    z_array = np.linspace(0, z_max, 1000)
    z_array = z_array[::-1]
    t_array = cosmo.age(z_array).to_value()

    return interp.interp1d(t_array, z_array, fill_value="extrapolate")
