import numpy as np

from typing import Union, Optional
from abc import ABC, abstractmethod
from astropy.table import Table
from astropy.cosmology import Cosmology


class PopulationBase(ABC):

    def __init__(self, cosmo: Cosmology):

        self.cosmo = cosmo

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def sample(self, seed: Optional[int] = None, n: Optional[int] = 1):
        pass


class GalaxyPopulation(PopulationBase):

    def __init__(
        self,
        cosmo: Cosmology,
        galaxy_table: Union[Table, dict],  # TODO: Check if alternative is dict or list
        kwargs_cut: dict,
    ):

        super().__init__(cosmo=cosmo)

        self.galaxy_table = galaxy_table
        self.kwargs_cut = kwargs_cut

    def __len__(self) -> int:
        return len(self.galaxy_table)

    @abstractmethod
    def sample(self, seed: Optional[int] = None, n: Optional[int] = 1):
        pass
