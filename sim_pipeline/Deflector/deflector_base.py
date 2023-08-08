from abc import ABC, abstractmethod

class DeflectorBase(ABC):
    """
    base class with functions all deflector classes must have to be able to render populations

    """
    def __init__(self, cosmo, sky_area):
        """

        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of solid angle.
        :type sky_area: `~astropy.units.Quantity`
        """
        self._cosmo = cosmo
        self._sky_area = sky_area


    @abstractmethod
    def deflector_position(self):
        """
        center of the deflector position

        :return: [x_pox, y_pos] in arc seconds
        """
        pass

    @abstractmethod
    def source_position(self):
        """
        source position, either the center of the extended source or the point source. If not present from the cataloge,
        it is drawn uniform within the circle of the test area centered on the lens

        :return: [x_pos, y_pos]
        """

    @abstractmethod
    def image_positions(self):
        """
        returns image positions by solving the lens equation, these are either the centers of the extended source, or
        the point sources in case of (added) point-like sources, such as quasars or SNe.

        :return: x-pos, y-pos
        """
        pass


    @abstractmethod
    def deflector_redshift(self):
        """

        :return: lens redshift
        """
        pass

    @abstractmethod
    def source_redshift(self):
        """

        :return: source redshift
        """
        pass

    @abstractmethod
    def einstein_radius(self):
        """
        Einstein radius, including the SIS + external convergence effect

        :return: Einstein radius [arc seconds]
        """
        pass

    @abstractmethod
    def deflector_ellipticity(self):
        """

        :return: e1_light, e2_light, e1_mass, e2_mass
        """
        pass

    def deflector_stellar_mass(self):
        """

        :return: stellar mass of deflector
        """
        pass

    def deflector_velocity_dispersion(self):
        """

        :return: velocity dispersion [km/s]
        """
        return self._lens_dict['vel_disp']

    def los_linear_distortions(self):
        """
        line-of-sight distortions in shear and convergence

        :return: kappa, gamma1, gamma2
        """
        pass

    @abstractmethod
    def deflector_magnitude(self, band):
        """
        apparent magnitude of the deflector for a given band

        :param band: imaging band
        :type band: string
        :return: magnitude of deflector in given band
        """
        pass
    
    @abstractmethod
    def point_source_magnitude(self, band, lensed=False):
        """
        point source magnitude, either unlensed (single value) or lensed (array) with macro-model magnifications

        # TODO: time-variability with time-delayed and micro-lensing

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: point source magnitude
        """
        pass
    
    @abstractmethod
    def extended_source_magnitude(self, band, lensed=False):
        """
        unlensed apparent magnitude of the extended source for a given band
        (assumes that size is the same for different bands)

        :param band: imaging band
        :type band: string
        :param lensed: if True, returns the lensed magnified magnitude
        :type lensed: bool
        :return: magnitude of source in given band
        """
        pass
    
    @abstractmethod
    def point_source_magnification(self):
        """
        macro-model magnification of point sources

        :return: signed magnification of point sources in same order as image positions
        """
        pass
    
    @abstractmethod
    def host_magnification(self):
        """
        compute the extended lensed surface brightness and calculates the integrated flux-weighted magnification factor
        of the extended host galaxy

        :return: integrated magnification factor of host magnitude
        """
        pass

    @abstractmethod
    def deflector_mass_model_lenstronomy(self):
        """
        returns lens model instance and parameters in lenstronomy conventions

        :return: lens_model_list, kwargs_lens
        """
        pass


    @abstractmethod
    def deflector_light_model_lenstronomy(self):
        """
        returns lens model instance and parameters in lenstronomy conventions

        :return: lens_model_list, kwargs_lens
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
    
    





