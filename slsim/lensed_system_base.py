from abc import ABC, abstractmethod
import numpy as np
from slsim.Sources.source import Source


class LensedSystemBase(ABC):
    """Abstract Base class to create a lens system with all lensing properties required
    to render populations."""

    def __init__(
        self,
        source_dict,
        deflector_dict,
        cosmo,
        test_area=4 * np.pi,
        variability_model=None,
        kwargs_variability=None,
    ):
        """
        :param source_dict: source properties
        :type source_dict: dict
        :param deflector_dict: deflector properties
        :type deflector_dict: dict
        :param variability_model: keyword for variability model to be used. This is an
         input for the Variability class.
        :type variability_model: str
        :param kwargs_variability: keyword arguments for the variability of a source.
         This is associated with an input for Variability class.
        :param cosmo: astropy.cosmology instance
        :param test_area: area (arc-sec^2) around lensing galaxy to be investigated

        """
        # self._source_dict = source_dict
        self.source = Source(source_dict, variability_model, kwargs_variability)
        self._deflector_dict = deflector_dict
        # TODO: tell them what keys the dictionary should contain
        self.test_area = test_area
        self.cosmo = cosmo

    @abstractmethod
    def deflector_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        pass

    @abstractmethod
    def source_position(self):
        """Source position, either the center of the extended source or the point
        source. If not present from the cataloge, it is drawn uniform within the circle
        of the test area centered on the lens.

        :return: [x_pos, y_pos] in arc seconds
        """
        pass

    @abstractmethod
    def image_positions(self):
        """Returns image positions by solving the lens equation, these are either the
        centers of the extended source, or the point sources in case of (added) point-
        like sources, such as quasars or SNe.

        :return: x-pos, y-pos
        """
        pass

    @abstractmethod
    def deflector_redshift(self):
        """Deflector redshift.

        :return: deflector redshift
        """
        pass

    @abstractmethod
    def source_redshift(self):
        """Source redshift.

        :return: source redshift
        """
        pass

    @abstractmethod
    def einstein_radius(self):
        """Einstein radius.

        :return: Einstein radius [arc seconds]
        """
        pass

    @abstractmethod
    def deflector_ellipticity(self):
        """Ellipticity components for deflector light and mass profile.

        :return: e1_light, e2_light, e1_mass, e2_mass
        """
        pass

    @abstractmethod
    def deflector_velocity_dispersion(self):
        """

        :return: velocity dispersion [km/s]
        """
        pass

    @abstractmethod
    def los_linear_distortions(self):
        """Line-of-sight distortions in shear and convergence.

        :return: kappa, gamma1, gamma2
        """
        pass

    @abstractmethod
    def deflector_magnitude(self, band):
        """Apparent magnitude of the deflector for a given band (AB mag)

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        pass

    @abstractmethod
    def point_source_magnitude(self, band, lensed=False):
        """Point source magnitude, either unlensed (single value) or lensed (array) with
        macro-model magnifications.

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: point source magnitude
        """
        pass

    @abstractmethod
    def extended_source_magnitude(self, band, lensed=False):
        """Apparent magnitude of the extended source for a given band (lensed or
        unlensed) (assumes that size is the same for different bands)

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        pass

    @abstractmethod
    def point_source_magnification(self):
        """Macro-model magnification of point sources.

        :return: signed magnification of point sources in same order as image positions
        """
        pass

    @abstractmethod
    def extended_source_magnification(self):
        """Extended source (or host) magnification.

        :return: integrated magnification factor of host magnitude
        """
        pass

    @abstractmethod
    def deflector_mass_model_lenstronomy(self):
        """Returns lens mass model instance and parameters in lenstronomy conventions.

        :return: lens_mass_model_list, kwargs_lens_mass
        """
        pass

    @abstractmethod
    def deflector_light_model_lenstronomy(self):
        """Returns lens model instance and parameters in lenstronomy conventions.

        :return: lens_light_model_list, kwargs_lens_light
        """
        pass

    @abstractmethod
    def source_light_model_lenstronomy(self):
        """Returns source light model instance and parameters in lenstronomy
        conventions.

        :return: source_light_model_list, kwargs_source_light
        """
        pass

    @abstractmethod
    def lenstronomy_kwargs(self, band=None):
        """

        :param band: imaging band, if =None, will result in un-normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions

        """
        pass
