from abc import ABC, abstractmethod
import warnings


class LensedPopulationBase(ABC):
    """Abstract Base Class to create a sample of lensed systems.

    All object that inherit from Lensed Sample must contain the methods
    it contains.
    """

    def __init__(
        self,
        sky_area=None,
        cosmo=None,
        use_jax=True,
    ):
        """

        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :type sky_area: `~astropy.units.Quantity`
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology instance
        :param use_jax: if True, will use JAX version of lenstronomy to do lensing calculations for models that are
         supported in JAXtronomy
        :type use_jax: bool
        """

        if sky_area is None:
            # sky_area = Quantity(value=0.1, unit="deg2")
            raise ValueError("No sky area provided. Please provide needed sky area.")
        self.sky_area = sky_area

        if cosmo is None:
            warnings.warn(
                "No cosmology provided, instead uses flat LCDM with default parameters"
            )
            from astropy.cosmology import FlatLambdaCDM

            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.cosmo = cosmo
        self._use_jax = use_jax

    @abstractmethod
    def select_lens_at_random(self):
        """Draw a random lens within the cuts of the lens and source, with
        possible additional cut in the lensing configuration.

        # as well as option to draw all lenses within the cuts within
        the area

        :return: Lens() instance with parameters of the deflector and
            lens and source light
        """
        pass

    @abstractmethod
    def deflector_number(self):
        """Number of potential deflectors (meaning all objects with mass that
        are being considered to have potential sources behind them)

        :return: number of potential deflectors
        """
        pass

    @abstractmethod
    def source_number(self):
        """Number of sources that are being considered to be placed in the sky
        area potentially aligned behind deflectors.

        :return: number of sources
        """
        pass

    @abstractmethod
    def draw_population(self, **kwargs):
        """Return full sample list of all lenses within the area.

        :return: List of LensedSystemBase instances with parameters of
            the deflectors and source.
        :rtype: list
        """

        pass
