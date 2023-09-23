import numpy as np
import numpy.random as random
from sim_pipeline.selection import deflector_cut
from abc import ABC, abstractmethod


class DeflectorBase(ABC):
    """Abstract Base Class to create a deflector object.

    All object that inherit from Lensed System must contain the methods it contains.
    """

    def __init__(self, deflector_table, kwargs_cut, cosmo, sky_area):
        """

        :param deflector_table: table with lens parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of solid angle.
        """
        self.deflector_table = deflector_table
        self.kwargs_cut = kwargs_cut
        self.cosmo = cosmo
        self.sky_area = sky_area

    @abstractmethod
    def deflector_number(self):
        """

        :return: number of deflectors
        """
        pass

    @abstractmethod
    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of deflector
        """
        pass
