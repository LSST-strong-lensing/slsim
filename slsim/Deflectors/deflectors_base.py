from abc import ABC, abstractmethod


class DeflectorsBase(ABC):
    """Abstract Base Class to create a class that accesses a set of deflectors.

    All object that inherit from Lensed System must contain the methods it contains.
    """

    def __init__(self, deflector_table, kwargs_cut, cosmo, sky_area):
        """

        :param deflector_table: table with lens parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        """
        self.deflector_table = deflector_table
        self.kwargs_cut = kwargs_cut
        self.cosmo = cosmo
        self.sky_area = sky_area

    @abstractmethod
    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        pass

    @abstractmethod
    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """
        pass
