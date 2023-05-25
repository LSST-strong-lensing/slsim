from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST


def kwargs_single_band(band, psf_type='GAUSSIAN', coadd_years=10):
    """
    this is the function for returning the band information

    :param band: string, 'u', 'g', 'r', 'i', 'z' or 'y' supported. Determines obs dictionary.
    :param psf_type: string, type of PSF ('GAUSSIAN' supported).
    :param coadd_years: int, number of years corresponding to num_exposures in obs dict. Currently supported: 1-10.
    """
    lsst = LSST(band=band, psf_type=psf_type, coadd_years=coadd_years)
    return lsst.kwargs_single_band()
