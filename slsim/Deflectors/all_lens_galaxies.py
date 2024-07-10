import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Deflectors.velocity_dispersion import vel_disp_abundance_matching
from slsim.Deflectors.elliptical_lens_galaxies import (
    elliptical_projected_eccentricity,
)
from slsim.Deflectors.deflectors_base import DeflectorsBase
from astropy.table import vstack


class AllLensGalaxies(DeflectorsBase):
    """Class describing all-type galaxies."""

    def __init__(
        self,
        red_galaxy_list,
        blue_galaxy_list,
        kwargs_cut,
        kwargs_mass2light,
        cosmo,
        sky_area,
    ):
        """
        :param red_galaxy_list: list of dictionary with elliptical galaxy
            parameters (supporting skypy pipelines)
        :type red_galaxy_list: astropy.Table
        :param blue_galaxy_list: list of dictionary with spiral galaxy
            parameters (supporting skypy pipelines)
        :type blue_galaxy_list: astropy.Table
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        """

        red_column_names = red_galaxy_list.colnames
        if "galaxy_type" not in red_column_names:
            red_galaxy_list["galaxy_type"] = "red"

        blue_column_names = blue_galaxy_list.colnames
        if "galaxy_type" not in blue_column_names:
            blue_galaxy_list["galaxy_type"] = "blue"

        galaxy_list = vstack([red_galaxy_list, blue_galaxy_list])

        super().__init__(
            deflector_table=galaxy_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )

        n = len(galaxy_list)
        column_names = galaxy_list.colnames
        if "vel_disp" not in column_names:
            galaxy_list["vel_disp"] = -np.ones(n)
        if "e1_light" not in column_names or "e2_light" not in column_names:
            galaxy_list["e1_light"] = -np.ones(n)
            galaxy_list["e2_light"] = -np.ones(n)
        if "e1_mass" not in column_names or "e2_mass" not in column_names:
            galaxy_list["e1_mass"] = -np.ones(n)
            galaxy_list["e2_mass"] = -np.ones(n)
        if "n_sersic" not in column_names:
            galaxy_list["n_sersic"] = -np.ones(n)

        galaxy_list = fill_table(galaxy_list)
        self._f_vel_disp = vel_disp_abundance_matching(
            galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
        )

        self._galaxy_select = deflector_cut(galaxy_list, **kwargs_cut)
        self._num_select = len(self._galaxy_select)
        self._galaxy_select["vel_disp"] = self._f_vel_disp(
            np.log10(self._galaxy_select["stellar_mass"])
        )
        # TODO: random reshuffle of matched list

    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        number = self._num_select
        return number

    def draw_deflector(self):
        """

        :return: dictionary of complete parameterization of a deflector
        """

        index = random.randint(0, self._num_select - 1)
        deflector = self._galaxy_select[index]
        if deflector["e1_light"] == -1 or deflector["e2_light"] == -1:
            e1_light, e2_light, e1_mass, e2_mass = elliptical_projected_eccentricity(
                **deflector
            )
            deflector["e1_light"] = e1_light
            deflector["e2_light"] = e2_light
            deflector["e1_mass"] = e1_mass
            deflector["e2_mass"] = e2_mass
        if deflector["n_sersic"] == -1:
            deflector["n_sersic"] = 4  # TODO make a better estimate with scatter
        return deflector


def fill_table(galaxy_list):
    """

    :param galaxy_list: ~Astropy.Table instance
    :return: table with additional columns
    """
    n = len(galaxy_list)
    column_names = galaxy_list.colnames
    if "vel_disp" not in column_names:
        galaxy_list["vel_disp"] = -np.ones(n)
    if "e1_light" not in column_names or "e2_light" not in column_names:
        galaxy_list["e1_light"] = -np.ones(n)
        galaxy_list["e2_light"] = -np.ones(n)
    if "e1_mass" not in column_names or "e2_mass" not in column_names:
        galaxy_list["e1_mass"] = -np.ones(n)
        galaxy_list["e2_mass"] = -np.ones(n)
    if "n_sersic" not in column_names:
        galaxy_list["n_sersic"] = -np.ones(n)
    return galaxy_list
