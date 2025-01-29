
def LSST_to_SDSS(lsst_u, lsst_g, lsst_r, lsst_i, lsst_z):

    '''
    Converts the five bands LSST magnitudes to the SDSS magnitudes
    using empirically derived transformations.
    These transformations are only valid for red galaxies and should not 
    be used for blue galaxies or other stellar populations. The coefficients 
    are based on polynomial fits to color terms.

    params:
    -- lsst_u : float
            u-band lsst magnitude
    -- lsst_g : float
            g-band lsst magnitude
    -- lsst_r : float
            r-band lsst magnitude
    -- lsst_i : float
            i-band lsst magnitude
    -- lsst_z : float
            z-band lsst magnitude    
    
    returns:   list
        A list of SDSS-equivalent magnitudes in the order [u, g, r, i, z].

    '''
    sdss_u  = lsst_u - 0.014285 +  (0.191787*(lsst_u-lsst_g))  +  (-0.062736*(lsst_u-lsst_g)**2)
    sdss_g  = lsst_g + 0.008059 +  (0.029470*(lsst_g-lsst_r))  +  (0.031589*(lsst_g-lsst_r)**2)
    sdss_r  = lsst_r - 0.001168 +  (0.017418*(lsst_r-lsst_i))  +  (0.021144*(lsst_r-lsst_i)**2)
    sdss_i  = lsst_i - 0.000026 +  (0.044532*(lsst_i-lsst_z))  +  (-0.013802*(lsst_i-lsst_z)**2)
    sdss_z  = lsst_z - 0.030518 +  (-0.206242*(lsst_i-lsst_z)) +  (0.084968*(lsst_i-lsst_z)**2)

    return [sdss_u, sdss_g, sdss_r, sdss_i, sdss_z]


def SDSS_to_LSST(sdss_u, sdss_g, sdss_r, sdss_i, sdss_z):

    '''
    Converts the five bands SDSS magnitudes to the LSST magnitudes
    using empirically derived transformations.
    These transformations are only valid for red galaxies and should not 
    be used for blue galaxies or other stellar populations. The coefficients 
    are based on polynomial fits to color terms.

    params:
    -- sdss_u : float
            u-band SDSS magnitude
    -- sdss_g : float
            g-band SDSS magnitude
    -- sdss_r : float
            r-band SDSS magnitude
    -- sdss_i : float
            i-band SDSS magnitude
    -- sdss_z : float
            z-band SDSS magnitude    
    
    returns:   list
        A list of LSST-equivalent magnitudes in the order [u, g, r, i, z].

    '''

    lsst_u  = sdss_u + 0.008274 +  (-0.128901*(sdss_u-sdss_g))  +  (0.021774*(sdss_u-sdss_g)**2)
    lsst_g  = sdss_g - 0.007948 +  (-0.025930*(sdss_g-sdss_r))  +  (-0.031108*(sdss_g-sdss_r)**2)
    lsst_r  = sdss_r + 0.001010 +  (-0.013431*(sdss_r-sdss_i))  +  (-0.024717*(sdss_r-sdss_i)**2)
    lsst_i  = sdss_i + 0.001282 +  (-0.039931*(sdss_i-sdss_z))  +  (0.011767*(sdss_i-sdss_z)**2)
    lsst_z  = sdss_z + 0.012325 + (0.191612*(sdss_i-sdss_z)) +  (-0.053284*(sdss_i-sdss_z)**2)

    return [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z]