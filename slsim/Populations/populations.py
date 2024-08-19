import warnings

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

    def _preprocess_galaxy_tables(
        self, galaxy_tables: Union[Table, list[Table]]
    ) -> Union[Table, list[Table]]:

        is_table = isinstance(galaxy_tables, Table)
        if is_table:
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

        # TODO START: MAKE THIS THE RESPONSIBILITY OF FUTURE CATALOG OBJECTS
        for galaxy_table, i in enumerate(galaxy_tables):
            galaxy_tables[i] = convert_to_slsim_convention(
                galaxy_catalog=galaxy_table,
                light_profile=self.light_profile,
                input_catalog_type=catalog_type,
            )

        for expected_column in expected_columns:
            for galaxy_table in galaxy_tables:
                column_names = galaxy_table.columns
                n_rows = len(galaxy_table)
                if expected_column not in column_names:
                    column = Column([-1] * n_rows, name=expected_column)
                    galaxy_table.add_column(column)
                if "ellipticity" not in column_names:
                    raise ValueError("ellipticity is missing in galaxy_table columns.")
        # TODO END

        # TODO: CONSIDER MAKING RESPONSIBILITY OF FUTURE CATALOG OBJECTS
        for galaxy_table in galaxy_tables:

            galaxy_table["vel_disp"] = np.where(
                galaxy_table["vel_disp"] == -1,
                vel_disp_from_m_star(galaxy_table["stellar_mass"]),
                galaxy_table["vel_disp"],
            )

            e1_light, e2_light, e1_mass, e2_mass = elliptical_projected_eccentricity(
                ellipticity=galaxy_table["ellipticity"],
            )
            galaxy_table["e1_light"] = np.where(
                galaxy_table["e1_light"] == -1, e1_light, galaxy_table["e1_light"]
            )
            galaxy_table["e2_light"] = np.where(
                galaxy_table["e2_light"] == -1, e2_light, galaxy_table["e2_light"]
            )
            galaxy_table["e1_mass"] = np.where(
                galaxy_table["e1_mass"] == -1, e1_mass, galaxy_table["e1_mass"]
            )
            galaxy_table["e2_mass"] = np.where(
                galaxy_table["e2_mass"] == -1, e2_mass, galaxy_table["e2_mass"]
            )

        if is_table:
            galaxy_tables = galaxy_tables[0]

        return galaxy_tables

    @abstractmethod
    def sample(self, seed: Optional[int] = None, n: Optional[int] = 1):
        pass


def epsilon2e(epsilon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Translates ellipticity definitions from.

    .. math::
        \epsilon = \\equic \\frac{1 - q^2}{1 + q^2}

    to

    .. math::
        e = \\equic \\frac{1-q}{1+q}

    Args:
        epsilon (Union[float, np.ndarray]): Ellipticity

    Raises:
        ValueError: If epsilon is not in the range [0, 1]

    Returns:
        Union[float, np.ndarray]: Eccentricity
    """

    is_valid = np.all((epsilon >= 0) & (epsilon <= 1))
    if not is_valid:
        raise ValueError("epsilon must be in the range [0, 1].")

    # Catch warnings from division by zero
    # since epsilon = 0 is a valid input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = (1 - np.sqrt(1 - epsilon**2)) / epsilon
        e = np.where(np.isnan(e), 0, e)

    return e


def elliptical_projected_eccentricity(
    ellipticity: Union[float, np.ndarray],
    light2mass_e_scaling: Union[float, np.ndarray] = 1.0,
    light2mass_e_scatter: Union[float, np.ndarray] = 0.1,
    light2mass_angle_scatter: Union[float, np.ndarray] = 0.1,
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> tuple[Union[float, np.ndarray]]:
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    Args:
        ellipticity (Union[float, np.ndarray]): Eccentricity amplitude (1-q^2)/(1+q^2).
        light2mass_e_scaling (Union[float, np.ndarray], optional): Scaling factor of mass eccentricity / light eccentricity. Defaults to 1.0.
        light2mass_e_scatter (Union[float, np.ndarray], optional): Scatter in light and mass eccentricities from the scaling relation. Defaults to 0.1.
        light2mass_angle_scatter (Union[float, np.ndarray], optional): Scatter in orientation angle between light and mass eccentricities. Defaults to 0.1.
        rng (Optional[np.random.Generator], optional): Numpy Random Generator instance. If None, use np.random. default_rng. Defaults to None.

    Raises:
        ValueError: If light2mass arguments do not have the same length as input ellipticity.

    Returns:
        tuple[Union[float, np.ndarray]]: e1_light, e2_light, e1_mass, e2_mass eccentricity components
    """

    if rng is None:
        rng = np.random.default_rng()

    ellipticity_is_float = isinstance(ellipticity, float)
    light2mass_are_float = (
        isinstance(light2mass_e_scaling, float)
        and isinstance(light2mass_e_scatter, float)
        and isinstance(light2mass_angle_scatter, float)
    )

    ellipticity = np.atleast_1d(ellipticity)
    light2mass_e_scaling = np.atleast_1d(light2mass_e_scaling)
    light2mass_e_scatter = np.atleast_1d(light2mass_e_scatter)
    light2mass_angle_scatter = np.atleast_1d(light2mass_angle_scatter)
    n = len(ellipticity)

    if light2mass_are_float:
        light2mass_e_scaling = np.full(n, light2mass_e_scaling)
        light2mass_e_scatter = np.full(n, light2mass_e_scatter)
        light2mass_angle_scatter = np.full(n, light2mass_angle_scatter)

    light2mass_args_valid = (
        n == len(light2mass_e_scaling)
        and n == len(light2mass_e_scatter)
        and n == len(light2mass_angle_scatter)
    )
    if not light2mass_args_valid:
        raise ValueError(
            "light2mass arguments must have the same length as input ellipticity."
        )

    e_light = epsilon2e(ellipticity)
    phi_light = rng.uniform(0, np.pi, size=n)
    e1_light = e_light * np.cos(2 * phi_light)
    e2_light = e_light * np.sin(2 * phi_light)

    e_mass = light2mass_e_scaling * ellipticity + rng.normal(
        loc=0, scale=light2mass_e_scatter, size=n
    )
    phi_mass = phi_light + rng.normal(loc=0, scale=light2mass_angle_scatter, size=n)
    e1_mass = e_mass * np.cos(2 * phi_mass)
    e2_mass = e_mass * np.sin(2 * phi_mass)

    if ellipticity_is_float:
        e1_light, e2_light, e1_mass, e2_mass = (
            e1_light[0],
            e2_light[0],
            e1_mass[0],
            e2_mass[0],
        )

    return e1_light, e2_light, e1_mass, e2_mass


def vel_disp_from_m_star(m_star: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the velocity dispersion of a galaxy from its stellar mass using an
    empirical power-law relation for elliptical galaxies.

    The power-law formula is given by:

    .. math::
        V_{\\mathrm{disp}} = 10^{2.32} \\left( \\frac{M_{\\mathrm{star}}}{10^{11}
        M_\\odot} \\right)^{0.24}

    Values taken from table 2 of [1]

    [1]:Auger, M. W., et al. "The Sloan Lens ACS Survey. X. Stellar, dynamical, and
    total mass correlations of massive elliptical galaxies." The Astrophysical
    Journal 724.1 (2010): 511.

    Args:
        m_star (Union[float, np.ndarray]): Stellar mass of the galaxy in solar masses.

    Returns:
        Union[float, np.ndarray]: Velocity dispersion in km/s.
    """

    v_disp = np.power(10, 2.32) * np.power(m_star / 1e11, 0.24)

    return v_disp
