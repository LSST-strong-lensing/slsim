from abc import ABC, abstractmethod
from slsim.LOS.los_individual import LOSIndividual
from slsim.Sources.source import Source


class LensedSystemBase(ABC):
    """Abstract Base class to create a lens system with all lensing properties
    required to render populations."""

    def __init__(
        self,
        source_class,
        deflector_class,
        los_class=None,
        multi_plane=None,
        shear=True,
        convergence=False,
    ):
        """
        :param source_class: :param source_class: A Source class instance or list of
         Source class instance
        :type source_class: Source class instance from slsim.Sources.source.
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
        :param los_class: Line of sight distortion class
        :type los_class: ~LOSIndividual instance
        :param multi_plane: None for single-plane, 'Source' for multi-source plane, 'Deflector' for multi-deflector plane,
            or 'Both' for both multi-deflector and multi-source plane
        :type multi_plane: None or str
        :param shear: whether to include external shear in multi-plane lensing
        :type shear: bool
        :param convergence: whether to include external convergence in multi-plane lensing
        :type convergence: bool
        """
        self.deflector = deflector_class
        self.multi_plane = multi_plane
        self.shear = shear
        self.convergence = convergence
        if isinstance(source_class, list):
            for source_class_ in source_class:
                assert isinstance(source_class_, Source)
            self._source = source_class
            # chose the highest redshift source to use conventionally use in lens
            #  mass model.
            self.max_redshift_source_class = max(
                self._source, key=lambda obj: obj.redshift
            )
            self.source_number = len(self._source)
            self._max_redshift_source_index = self._source.index(
                self.max_redshift_source_class
            )
        else:
            assert isinstance(source_class, Source)
            self._source = [source_class]
            self.source_number = 1
            # this is for single source case. self.max_redshift_source_class and
            # self.source are the same class. The difference is only that one is in the
            #  form of list and other is just a Source instance. This is done just for
            # the completion of routine to make things consistent in both single source
            # and double source case.
            self.max_redshift_source_class = source_class
            self._max_redshift_source_index = 0

        if isinstance(source_class, list):
            self._source = source_class
        else:
            self._source = [source_class]
        if los_class is None:
            los_class = LOSIndividual()
        self.los_class = los_class

    @abstractmethod
    def deflector_position(self):
        """Center of the deflector position.

        :return: [x_pox, y_pos] in arc seconds
        """
        pass

    @abstractmethod
    def extended_source_image_positions(self):
        """Returns extended source image positions by solving the lens
        equation.

        :return: x-pos, y-pos
        """
        pass

    @abstractmethod
    def point_source_image_positions(self):
        """Returns point source image positions by solving the lens equation.
        In the absence of a point source, this function returns the solution
        for the center of the extended source.

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
    def source_redshift_list(self):
        """Source redshift.

        :return: list of each source redshift
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
        """Point source magnitude, either unlensed (single value) or lensed
        (array) with macro-model magnifications.

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: point source magnitude
        """
        pass

    @abstractmethod
    def extended_source_magnitude(self, band, lensed=False):
        """Apparent magnitude of the extended source for a given band (lensed
        or unlensed) (assumes that size is the same for different bands)

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

        :return: signed magnification of point sources in same order as
            image positions
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
        """Returns lens mass model instance and parameters in lenstronomy
        conventions.

        :return: lens_mass_model_list, kwargs_lens_mass
        """
        pass

    @abstractmethod
    def deflector_light_model_lenstronomy(self, band):
        """Returns lens model instance and parameters in lenstronomy
        conventions.

        :param band: imaging band
        :type band: str
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
