from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid


def kwargs_single_band(band, observatory="LSST", **kwargs):
    """This is the function for returning the band information.

    :param band: 'u', 'g', 'r', 'i', 'z' or 'y' supported. Determines
        imaging bands.
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
    elif observatory == "Euclid":
        observatory = Euclid(band=band, **kwargs)
    return observatory.kwargs_single_band()


def get_observatory(band):
    """Determine the observatory based on the imaging band.

    :param band: imaging band name
    :type band: str
    :return: observatory name ("LSST", "Roman", or "Euclid")
    :rtype: str
    :raises ValueError: if band is not recognized for any observatory

    Supported bands:
        - LSST: 'u', 'g', 'r', 'i', 'z', 'y'
        - Roman: 'F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F146'
        - Euclid: 'VIS'
    """
    if band in ["u", "g", "r", "i", "z", "y"]:
        return "LSST"
    elif band in ["F062", "F087", "F106", "F129", "F158", "F184", "F146"]:
        return "Roman"
    elif band in ["VIS"]:
        return "Euclid"
    else:
        raise ValueError(f"Band {band} not recognized for any observatory.")
