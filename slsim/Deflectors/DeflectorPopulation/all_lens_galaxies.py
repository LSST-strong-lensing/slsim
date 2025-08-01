import numpy as np
import numpy.random as random
from slsim.Lenses.selection import object_cut
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Deflectors.DeflectorPopulation.elliptical_lens_galaxies import (
    elliptical_projected_eccentricity,
)
from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase
from astropy.table import vstack
from slsim.Util.param_util import catalog_with_angular_size_in_arcsec
from slsim.Deflectors.deflector import Deflector


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
        gamma_pl=None,
        catalog_type="skypy",
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
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation
         of a density slope for gaussian distribution or minimum and maximum values of
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        :param catalog_type: type of the catalog. If user is using deflector catalog
         other than generated from skypy pipeline, we require them to provide angular
         size of the galaxy in arcsec and specify catalog_type as None. Otherwise, by
         default, this class considers deflector catalog is generated using skypy
         pipeline.
        :type catalog_type: str. "skypy" or None.
        """
        red_galaxy_list = catalog_with_angular_size_in_arcsec(
            galaxy_catalog=red_galaxy_list, input_catalog_type=catalog_type
        )
        blue_galaxy_list = catalog_with_angular_size_in_arcsec(
            galaxy_catalog=blue_galaxy_list, input_catalog_type=catalog_type
        )
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
            gamma_pl=gamma_pl,
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

        self._galaxy_select = object_cut(galaxy_list, **kwargs_cut)
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
        deflector_class = Deflector(deflector_type=self.deflector_profile, **deflector)
        return deflector_class


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
