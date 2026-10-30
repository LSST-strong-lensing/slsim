import numpy as np
import speclite.filters

from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid

_OBSERVATORY_REGISTRY = {}

# Default options in SLSim
ROMAN_BAND_LIST = ["F062", "F087", "F106", "F129", "F158", "F184", "F146", "F213"]
LSST_BAND_LIST = ["u", "g", "r", "i", "z", "y"]
EUCLID_BAND_LIST = ["VIS", "Y", "J", "H"]

# Ancillary bandpasses that are used by catalog sources but are not imaging
# observatories registered below.
_ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON = {"F814W": 0.805}


def check_speclite_name(band):
    """Checks if the raw band name is a valid speclite filter.

    Returns the band name if valid, otherwise returns None. This will
    serve as the default speclite_fmt for observatories that use the
    same band names as their speclite filters.
    """
    try:
        # attempt to load the filter from speclite's registry
        speclite.filters.load_filter(band)
        return band
    except ValueError:
        # speclite doesn't recognize this exact name
        return None


def register_observatory(
    name: str,
    observatory_class,
    bands: list,
    speclite_fmt=check_speclite_name,
    sncosmo_fmt=None,
    effective_wavelengths=None,
):
    """Register a new observatory to integrate it with image simulation tools.

    This allows external or user-defined observatories (e.g., "MidEx")
    to be automatically recognized by functions like ``kwargs_single_band``.

    :param name: The identifier for the observatory (e.g., "MidEx").
    :type name: str
    :param observatory_class: The class defining the observatory's configuration.
        Similar to the ``lenstronomy.SimulationAPI.ObservationConfig`` classes,
        this class must fulfill two requirements:
        1. Its constructor must accept ``band`` as a keyword argument.
        2. It must expose a ``kwargs_single_band()`` method that returns a dictionary of lenstronomy observation parameters.
    :type observatory_class: type
    :param bands: List of band name strings owned by this observatory. E.g., for LSST this would be ['u', 'g', 'r', 'i', 'z', 'y'].
    :type bands: list[str]
    :param speclite_fmt: A callable function that takes a ``band`` string and returns the corresponding
        speclite filter name (e.g., ``lambda b: f"MidEx-{b}"``). Set to ``None`` if the
        observatory does not utilize speclite filters.
    :type speclite_fmt: callable, optional
    :param sncosmo_fmt: A callable function that takes a ``band`` string and returns the corresponding
        sncosmo bandpass name. Set to ``None`` to use the raw band name as the sncosmo bandpass name.
    :type sncosmo_fmt: callable, optional
    :param effective_wavelengths: Optional mapping from registered band names to
        throughput-weighted effective wavelengths in microns. Use this only
        when the responses are not available through ``speclite_fmt``.
    :type effective_wavelengths: dict or None

    Given below is a simple example of how to define a custom observatory and register it using this function.
    A sophisticated example demonstrating full image simulation capabilities can be found at https://github.com/timedilatesme/MidEx-sims/blob/main/v1/lagn_sims.ipynb

    Example:
    --------
    .. code-block:: python

        import copy
        import lenstronomy.Util.util as util
        import slsim.simulation.image_quality_lenstronomy as iql

        # Specify bands and their corresponding observation parameters for the custom observatory
        custom_band_obs = {
            "A": {"exposure_time": 90.0, "sky_brightness": 22.0, "magnitude_zero_point": 26.0},
            "B": {"exposure_time": 90.0, "sky_brightness": 21.5, "magnitude_zero_point": 25.8}
        }

        # Define the observatory class
        class CustomObs(object):
            \"\"\"Class containing CustomObs instrument and observation configurations.\"\"\"

            def __init__(self, band="A", **kwargs):
                if band not in custom_band_obs:
                    raise ValueError(f"Band '{band}' not supported! Choose from {list(custom_band_obs.keys())}.")

                self.obs = copy.deepcopy(custom_band_obs[band])
                self.camera = {
                    "read_noise": 5.0,
                    "pixel_scale": 0.15,
                    "ccd_gain": 2.0,
                }

            def kwargs_single_band(self):
                return util.merge_dicts(self.camera, self.obs)

        # Register the new observatory
        iql.register_observatory(
            name="CustomObs",
            observatory_class=CustomObs,
            bands=list(_custom_band_obs.keys()),
            speclite_fmt=lambda b: f"CustomObs-{b}",
            sncosmo_fmt=lambda b: f"customobs{b}"
        )
    """
    _OBSERVATORY_REGISTRY[name] = {
        "class": observatory_class,
        "bands": list(bands),
        "speclite_fmt": speclite_fmt,
        "sncosmo_fmt": sncosmo_fmt,
        "effective_wavelengths": dict(effective_wavelengths or {}),
    }


# Pre-registered observatories (LSST, Roman, and Euclid)
register_observatory(
    name="LSST",
    observatory_class=LSST,
    bands=LSST_BAND_LIST,
    speclite_fmt=lambda band: f"lsst2023-{band}",
    sncosmo_fmt=lambda band: f"lsst{band}",
)
register_observatory(
    name="Roman",
    observatory_class=Roman,
    bands=ROMAN_BAND_LIST,
    speclite_fmt=lambda band: f"Roman-{band}",
    sncosmo_fmt=lambda band: f"{band}",
    effective_wavelengths={
        "F062": 0.620,
        "F087": 0.870,
        "F106": 1.060,
        "F129": 1.290,
        "F146": 1.460,
        "F158": 1.580,
        "F184": 1.840,
        "F213": 2.130,
    },
)
register_observatory(
    name="Euclid",
    observatory_class=Euclid,
    bands=EUCLID_BAND_LIST,
    speclite_fmt=lambda band: f"Euclid-{band}",
    sncosmo_fmt=lambda band: f"euclid{band}",
)


def _get_observatory_name_for_band(band):
    """Return the observatory name that owns *band*, searching the registry.

    :param band: Imaging band name.
    :raises ValueError: if no registered observatory claims the band.
    """
    for obs_name, info in _OBSERVATORY_REGISTRY.items():
        if band in info["bands"]:
            return obs_name
    raise ValueError(
        f"Band '{band}' is not recognised by any registered observatory. "
        f"Registered bands: { {o: i['bands'] for o, i in _OBSERVATORY_REGISTRY.items()} }"
    )


def get_observatory(band):
    """Determine the observatory based on the imaging band.

    Queries the registry; works for any registered observatory.

    :param band: Imaging band name.
    :raises ValueError: if the band does not belong to any observatory.
    """
    return _get_observatory_name_for_band(band)


def kwargs_single_band(band, observatory=None, **kwargs):
    """Return the lenstronomy single-band keyword dict for a given band.

    :param band: Imaging band name (e.g. ``'g'``, ``'F062'``, ``'VIS'``, etc.).
    :type band: str
    :param observatory: Observatory name.  When ``None`` the observatory registry is
        queried automatically based on *band*.
    :type observatory: str or None
    :param kwargs: Additional keyword arguments forwarded to the observatory
        class constructor (e.g. ``coadd_years``).
    :return: Configuration dict of imaging data for lenstronomy.
    :rtype: dict
    """
    if observatory is None:
        observatory = get_observatory(band)

    if observatory not in _OBSERVATORY_REGISTRY:
        raise ValueError(
            f"Observatory '{observatory}' is not registered. "
            f"Registered observatories: {list(_OBSERVATORY_REGISTRY.keys())}"
        )

    obs_class = _OBSERVATORY_REGISTRY[observatory]["class"]
    obs_instance = obs_class(band=band, **kwargs)
    return obs_instance.kwargs_single_band()


def get_speclite_filtername(band):
    """Get the speclite filter name corresponding to the given band.

    :param band: imaging band name
    :type band: str
    :return: speclite filter name
    :rtype: str
    :raises ValueError: if the band is not registered or has no speclite
        filter.

    Default Supported bands:
        - LSST: 'u', 'g', 'r', 'i', 'z', 'y'
        - Roman: 'F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F146', 'F213'
        - Euclid: 'VIS', 'Y', 'J', 'H'
    """
    obs_name = get_observatory(band)
    fmt = _OBSERVATORY_REGISTRY[obs_name]["speclite_fmt"]
    if fmt is None:
        raise ValueError(
            f"Observatory '{obs_name}' (band '{band}') has no speclite filter registered."
        )
    return fmt(band)


def get_speclite_filternames(bands):
    """Get a list of speclite filter names corresponding to the provided bands.

    :param bands: list of imaging band names. E.g., ['u', 'g', 'r', 'F062', 'VIS'].
    :type bands: list of str
    :return: list of speclite filter names in the same order as input bands
    :rtype: list of str
    :raises ValueError: if any band is not recognized for any observatory or has no speclite
        filter.

    Supported bands:
        - LSST: 'u', 'g', 'r', 'i', 'z', 'y'
        - Roman: 'F062', 'F087', 'F106', 'F129', 'F158', 'F184', 'F146', 'F213'
        - Euclid: 'VIS', 'Y', 'J', 'H'
    """
    return [get_speclite_filtername(band) for band in bands]


def get_sncosmo_filtername(band):
    """Get the sncosmo bandpass name corresponding to the given band.

    :param band: imaging band name
    :type band: str
    :return: sncosmo bandpass name
    :rtype: str
    """
    obs_name = get_observatory(band)
    fmt = _OBSERVATORY_REGISTRY[obs_name].get("sncosmo_fmt")

    # If no specific sncosmo_fmt was registered, fallback to just the raw band name
    if fmt is None:
        return band

    return fmt(band)


def get_all_supported_bands():
    """Return every band name currently registered across all observatories.

    :return: Flat list of band name strings.
    :rtype: list of str
    """
    all_bands = []
    for info in _OBSERVATORY_REGISTRY.values():
        all_bands.extend(info["bands"])
    return all_bands


def get_band_effective_wavelength(band):
    """Return the throughput-weighted effective wavelength of a band.

    The registered speclite response is used when available. Roman and
    HST bands that are not shipped by speclite use explicitly configured
    fallback values. Band-list position is deliberately not used as a
    wavelength proxy.

    :param band: Imaging band name.
    :type band: str
    :return: Effective wavelength in microns.
    :rtype: float
    :raises ValueError: if the band is not registered.
    """
    if band in _ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON:
        return _ADDITIONAL_BAND_EFFECTIVE_WAVELENGTH_MICRON[band]

    obs_name = get_observatory(band)
    configured_wavelengths = _OBSERVATORY_REGISTRY[obs_name]["effective_wavelengths"]
    if band in configured_wavelengths:
        return float(configured_wavelengths[band])

    filter_name = get_speclite_filtername(band)
    if filter_name is None:
        raise ValueError(
            f"Band '{band}' has neither an explicitly configured effective "
            f"wavelength nor a registered speclite filter response."
        )
    response = speclite.filters.load_filter(filter_name)
    return float(response.effective_wavelength.to("micron").value)


def get_band_central_wavelength(band):
    """Backward-compatible alias for :func:`get_band_effective_wavelength`."""
    return get_band_effective_wavelength(band)


def get_band_log_wavelength_ratio(band, reference_band):
    """Return ``log(lambda_band / lambda_reference)``.

    Both wavelengths are throughput-weighted effective wavelengths. This
    is the chromatic coordinate used by the local power-law SED
    approximation.
    """
    wavelength = get_band_effective_wavelength(band)
    reference_wavelength = get_band_effective_wavelength(reference_band)
    if wavelength <= 0 or reference_wavelength <= 0:
        raise ValueError("Band effective wavelengths must be positive.")
    return float(np.log(wavelength / reference_wavelength))
