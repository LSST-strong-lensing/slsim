from abc import ABC, abstractmethod


class SourcePopBase(ABC):
    """Base class with functions all source classes must have to be able to render
    populations."""

    def __init__(
        self,
        cosmo,
        sky_area,
        variability_model=None,
        kwargs_variability_model=None,
        agn_driving_variability_model=None,
        agn_driving_kwargs_variability=None,
    ):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param variability_model: keyword for the variability model to be used. This is
         a population argument, not the light curve parameter for the individual
         sources.
        :param kwargs_variability_model: keyword arguments for the variability of
         a source. This is a population argument, not the light curve parameter for
         the individual sources.
        :param agn_driving_variability_model: Variability model with light_curve output
         which drives the variability across all bands of the agn. eg: "light_curve",
         "sinusoidal", "bending_power_law"
        :param agn_driving_kwargs_variability: Dictionary containing agn variability
         parameters for the driving variability class. eg: variable_agn_kwarg_dict =
         {"length_of_light_curve": 1000, "time_resolution": 1,
         "log_breakpoint_frequency": 1 / 20, "low_frequency_slope": 1,
         "high_frequency_slope": 3, "normal_magnitude_variance": 0.1}. For the detailed
          explanation of these parameters, see generate_signal() function in
          astro_util.py.
        """
        self.source_type = None
        self.sky_area = sky_area
        self._cosmo = cosmo
        self._variab_model = variability_model
        self._kwargs_variab_model = kwargs_variability_model
        self.agn_driving_variability_model = agn_driving_variability_model
        self.agn_driving_kwargs_variability = agn_driving_kwargs_variability

    @property
    @abstractmethod
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        pass

    @property
    @abstractmethod
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        pass

    @abstractmethod
    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """
        pass

    @property
    def variability_model(self):
        """
        :return: keyword for the variability model
        """
        return self._variab_model

    @property
    def kwargs_variability(self):
        """
        :return: dict of keyword arguments for the variability model.
        """
        return self._kwargs_variab_model
