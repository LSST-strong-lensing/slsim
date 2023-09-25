import numpy as np

"This module aims to have realistic variability models for AGN and Supernovae."

def sinusoidal_variability(x):
    "This function provides sinosoidal variability as a function of time. This is used for the prototype code."
    
    """
    :param x: observation time
    :return: variability for a given time
    """
    amplitude = 1
    frequency = 0.5
    return amplitude*np.sin(2*np.pi*frequency*x)