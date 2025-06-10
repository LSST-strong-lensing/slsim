import numpy as np


def convergence_mean_0(kappa_data):
    """Adjusts the input kappa data by subtracting the mean of non-zero
    elements from the non-zero values only (above a small threshold to handle
    floating-point precision). Keeps zeros as is. Returns the adjusted data in
    the same format (list or numpy array) as the input.

    :param kappa_data: The input kappa data to be adjusted.
    :type kappa_data: list or numpy.ndarray
    :return: The adjusted kappa data in the same format as the input.
    :rtype: list or numpy.ndarray
    """

    input_type = type(kappa_data)
    kappa_array = np.array(kappa_data)
    non_zero_mask = np.abs(kappa_array) > 1e-6
    mean_kappa_non_zero = np.mean(kappa_array[non_zero_mask])
    adjusted_kappa_array = np.where(
        non_zero_mask, kappa_array - mean_kappa_non_zero, kappa_array
    )
    if input_type is list:
        return adjusted_kappa_array.tolist()
    else:
        return adjusted_kappa_array
