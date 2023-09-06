from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman


def kwargs_single_band(band, observatory="LSST", **kwargs):
    """This is the function for returning the band information.

    :param band: 'u', 'g', 'r', 'i', 'z' or 'y' supported. Determines imaging bands.
    :type band: str
    :param observatory: observatory chosen
    :type observatory: str
    :param kwargs: additional keyword arguments for the bands
    :type kwargs: dict
    :return: configuration of imaging data
    :rtype: dict
    """
    if observatory == "LSST":
        observatory = LSST(band=band, **kwargs)
    elif observatory == "Roman":
        observatory = Roman(band=band, **kwargs)
    return observatory.kwargs_single_band()
