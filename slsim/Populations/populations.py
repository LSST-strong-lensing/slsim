import numpy as np

from slsim.Util import param_util
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

        is_table = isinstance(galaxy_table, Table)
        is_list = isinstance(galaxy_table, list)
        if is_list:
            containts_only_tables = all(
                isinstance(table, Table) for table in galaxy_table
            )
        if not (is_table or is_list and containts_only_tables):
            raise ValueError(
                "galaxy_table must be an astropy table or a list of astropy tables."
            )

    def __len__(self) -> int:
        return len(self.galaxy_table)

    def _preprocess_galaxy_table(
        self, galaxy_tables: Union[Table, list[Table]]
    ) -> Union[Table, list[Table]]:

        n = len(galaxy_tables)
        column_names = galaxy_tables.colnames
        if isinstance(galaxy_tables, Table):
            galaxy_tables = [galaxy_tables]

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


def elliptical_projected_eccentricity(
    ellipticity,
    light2mass_e_scaling=1,
    light2mass_e_scatter=0.1,
    light2mass_angle_scatter=0.1,
    **kwargs
):
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    :param ellipticity: eccentricity amplitude (1-q^2)/(1+q^2)
    :type ellipticity: float [0,1)
    :param light2mass_e_scaling: scaling factor of mass eccentricity / light
        eccentricity
    :param light2mass_e_scatter: scatter in light and mass eccentricities from the
        scaling relation
    :param light2mass_angle_scatter: scatter in orientation angle between light and mass
        eccentricity
    :param kwargs: deflector properties
    :type kwargs: dict
    :return: e1_light, e2_light,e1_mass, e2_mass eccentricity components
    """
    e_light = param_util.epsilon2e(ellipticity)
    phi_light = np.random.uniform(0, np.pi)
    e1_light = e_light * np.cos(2 * phi_light)
    e2_light = e_light * np.sin(2 * phi_light)
    e_mass = light2mass_e_scaling * ellipticity + np.random.normal(
        loc=0, scale=light2mass_e_scatter
    )
    phi_mass = phi_light + np.random.normal(loc=0, scale=light2mass_angle_scatter)
    e1_mass = e_mass * np.cos(2 * phi_mass)
    e2_mass = e_mass * np.sin(2 * phi_mass)
    return e1_light, e2_light, e1_mass, e2_mass


def vel_disp_from_m_star(m_star):
    """Function to calculate the velocity dispersion from the staller mass using
    empirical relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::

         V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
         M_\\odot} \\right)^{0.24}

    2.32,0.24 is the parameters from [1] table 2
    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    :param m_star: stellar mass in the unit of solar mass
    :return: the velocity dispersion ("km/s")
    """
    v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)
    return v_disp
