import numpy as np

from typing import Union, Optional
from abc import ABC, abstractmethod
from astropy.table import Table, Column
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
        galaxy_table: Union[Table, list[Table]],
        kwargs_cut: dict,
        light_profile: str = "single_sersic",
    ):

        super().__init__(cosmo=cosmo)

        self.galaxy_table = galaxy_table
        self.kwargs_cut = kwargs_cut
        self.light_profile = light_profile

        if light_profile not in ["single_sersic", "double_sersic"]:
            raise ValueError(
                "light_profile %s not supported. Choose between single_sersic or double_sersic."
                % light_profile
            )

    def __len__(self) -> int:
        return len(self.galaxy_table)

    def _preprocess_galaxy_table(
        self, galaxy_table: Union[Table, list[Table]]
    ) -> Union[Table, list[Table]]:

        n = len(galaxy_table)
        column_names = galaxy_table.colnames
        is_table = isinstance(galaxy_table, Table)
        is_list = isinstance(galaxy_table, list)
        if is_list:
            containts_only_tables = all(
                isinstance(table, Table) for table in galaxy_table
            )

        if is_table:
            galaxy_tables = [galaxy_table]
        elif is_list and containts_only_tables:
            galaxy_tables = galaxy_table
        else:
            raise ValueError(
                "galaxy_table must be an astropy table or a list of astropy tables."
            )

        expected_columns = [
            "e1",
            "e2",
            "n_sersic",
            "n_sersic_0",
            "n_sersic_1",
            "e0_1",
            "e0_2",
            "e1_1",
            "e1_2",
            "angular_size0",
            "angular_size1",
        ]

        for galaxy_table, i in enumerate(galaxy_tables):
            galaxy_tables[i] = convert_to_slsim_convention(
                galaxy_catalog=galaxy_table,
                light_profile=self.light_profile,
                input_catalog_type=catalog_type,
            )

        for expected_column in expected_columns:
            for galaxy_table in galaxy_tables:
                if expected_column not in column_names:
                    column = Column([-1] * n, name=expected_column)
                    galaxy_table.add_column(column)
                if "ellipticity" not in column_names:
                    raise ValueError("ellipticity is missing in galaxy_table columns.")

        if is_table:
            galaxy_tables = galaxy_tables[0]

        return galaxy_tables

    @abstractmethod
    def sample(self, seed: Optional[int] = None, n: Optional[int] = 1):
        pass
