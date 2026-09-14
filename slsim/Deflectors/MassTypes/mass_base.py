from abc import ABC, abstractmethod


class MassBase(ABC):
    """Class of a single deflector with quantities only related to the
    deflector (independent of the source)"""

    def __init__(
        self,
        light,
        vel_disp=None,
    ):
        """

        :param light: light model class
        :type light: ~slsim.Sources.source.Source() class
        :param vel_disp: velocity dispersion [km/s]
        :type vel_disp: float or None
        """

        self._vel_disp = vel_disp
        self._light = light
        self.num_mass_models = 1

    @abstractmethod
    def velocity_dispersion(self, cosmo=None):
        """Velocity dispersion of deflector.

        :param cosmo: ~astropy.cosmology class
        :return: velocity dispersion [km/s]
        """
        return self._vel_disp

    @abstractmethod
    def mass_model_lenstronomy(self, lens_cosmo=None, spherical=False):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param lens_cosmo: lens cosmology model
        :type lens_cosmo: ~lenstronomy.Cosmo.LensCosmo instance
        :param spherical: if True, makes spherical assumption
        :type spherical: bool
        :return: lens_mass_model_list, kwargs_lens_mass
        """
        pass

    @property
    def mass_properties(self):
        """Mass density logarithmic slope.

        :return: gamma (with =2 is isothermal), velocity dispersion
            [km/s]
        :rtype: dict
        """
        return {}

    @property
    @abstractmethod
    def ellipticity(self):
        """Deflector eccentricities.

        :return: e1, e2
        """
        pass
