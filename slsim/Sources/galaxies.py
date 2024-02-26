import numpy as np
import numpy.random as random
from slsim.selection import deflector_cut
from slsim.Util import param_util
from slsim.Sources.source_pop_base import SourcePopBase
from astropy.table import Column
from slsim.Util.param_util import average_angular_size, axis_ratio, eccentricity
from lenstronomy.Util import constants


class Galaxies(SourcePopBase):
    """Class describing elliptical galaxies."""

    def __init__(
        self,
        galaxy_list,
        kwargs_cut,
        cosmo,
        sky_area,
        light_profile="single_sersic",
        list_type="astropy_table",
        catalog_type=None,
    ):
        """

        :param galaxy_list: list of dictionary with galaxy parameters
        :type galaxy_list: astropy Table object
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :param light_profile: keyword for number of sersic profile to use in source
         light model. accepted kewords: "single_sersic", "double_sersic".
        :param list_type: format of the source catalog file. Currently, it supports a
         single astropy table or a list of astropy tables.
        :type sky_area: `~astropy.units.Quantity`
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it.
        :type catalog_type: str. eg: "scotch" or None
        """
        super(Galaxies, self).__init__(cosmo=cosmo, sky_area=sky_area)
        self.n = len(galaxy_list)
        self.light_profile = light_profile
        # add missing keywords in astropy.Table object
        if list_type == "astropy_table":
            galaxy_list = convert_to_slsim_convention(
                galaxy_catalog=galaxy_list,
                light_profile=self.light_profile,
                input_catalog_type=catalog_type,
            )
            column_names_update = galaxy_list.colnames
            if light_profile == "single_sersic":
                if "e1" not in column_names_update or "e2" not in column_names_update:
                    galaxy_list["e1"] = -np.ones(self.n)
                    galaxy_list["e2"] = -np.ones(self.n)
                if "n_sersic" not in column_names_update:
                    galaxy_list["n_sersic"] = -np.ones(self.n)
            if light_profile == "double_sersic":
                # these are the name convention for double sersic profiles.
                if (
                    "n_sersic_0" not in column_names_update
                    or "n_sersic_1" not in column_names_update
                ):
                    galaxy_list["n_sersic_0"] = -np.ones(self.n)
                    galaxy_list["n_sersic_1"] = -np.ones(self.n)
                if (
                    "e0_1" not in column_names_update
                    or "e0_2" not in column_names_update
                ):
                    galaxy_list["e0_1"] = -np.ones(self.n)
                    galaxy_list["e0_2"] = -np.ones(self.n)
                if (
                    "e1_1" not in column_names_update
                    or "e1_2" not in column_names_update
                ):
                    galaxy_list["e1_1"] = -np.ones(self.n)
                    galaxy_list["e1_2"] = -np.ones(self.n)
                if (
                    "angular_size0" not in column_names_update
                    or "angular_size1" not in column_names_update
                ):
                    galaxy_list["angular_size0"] = -np.ones(self.n)
                    galaxy_list["angular_size1"] = -np.ones(self.n)
        else:
            column_names = galaxy_list[0].colnames
            if "ellipticity" not in column_names:
                raise ValueError("ellipticity is missing in galaxy_list columns.")
            if "e1" not in column_names or "e2" not in column_names:
                for table in galaxy_list:
                    new_column_length = len(table)
                    new_column_1 = Column([-1.0] * new_column_length, name="e1")
                    new_column_2 = Column([-1.0] * new_column_length, name="e2")
                    table.add_columns([new_column_1, new_column_2])
            if "n_sersic" not in column_names:
                for table in galaxy_list:
                    new_column_length = len(table)
                    new_column = Column([-1] * new_column_length, name="n_sersic")
                    table.add_column(new_column)
        # make cuts
        self._galaxy_select = deflector_cut(
            galaxy_list, list_type=list_type, **kwargs_cut
        )
        self._num_select = len(self._galaxy_select)
        self.list_type = list_type

    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        number = self.n
        return number

    def draw_source(self):
        """Choose source at random.

        :return: dictionary of source
        """

        index = random.randint(0, self._num_select - 1)
        galaxy = self._galaxy_select[index]
        if "a_rot" in galaxy.colnames:
            phi_rot = galaxy["a_rot"]
        else:
            phi_rot = None
        if self.light_profile == "single_sersic":
            if "ellipticity" in galaxy.colnames:
                if galaxy["e1"] == -1 or galaxy["e2"] == -1:
                    e1, e2 = galaxy_projected_eccentricity(
                        float(galaxy["ellipticity"]), rotation_angle=phi_rot
                    )
                    galaxy["e1"] = e1
                    galaxy["e2"] = e2
            else:
                raise ValueError("ellipticity is missing in galaxy_list columns.")
            if galaxy["n_sersic"] == -1:
                galaxy["n_sersic"] = 1  # TODO make a better estimate with scatter
        elif self.light_profile == "double_sersic":
            if galaxy["e0_1"] == -1 or galaxy["e0_2"] == -1:
                if "ellipticity0" in galaxy.colnames:
                    e0_1, e0_2 = galaxy_projected_eccentricity(
                        float(galaxy["ellipticity0"]), rotation_angle=phi_rot
                    )
                    galaxy["e0_1"] = e0_1
                    galaxy["e0_2"] = e0_2
                elif "a0" in galaxy.colnames and "b0" in galaxy.colnames:
                    axis_ratio_0 = axis_ratio(a=galaxy["a0"], b=galaxy["b0"])
                    ellip_0 = eccentricity(q=axis_ratio_0)
                    e0_1, e0_2 = galaxy_projected_eccentricity(
                        float(ellip_0), rotation_angle=phi_rot
                    )
                    galaxy["e0_1"] = e0_1
                    galaxy["e0_2"] = e0_2
                else:
                    raise ValueError(
                        "ellipticity or semi-major and semi-minor axis are missing for"
                        "the first light profile in galaxy_list columns"
                    )

            if galaxy["e1_1"] == -1 or galaxy["e1_2"] == -1:
                if "ellipticity1" in galaxy.colnames:
                    e1_1, e1_2 = galaxy_projected_eccentricity(
                        float(galaxy["ellipticity1"]), rotation_angle=phi_rot
                    )
                    galaxy["e1_1"] = e1_1
                    galaxy["e1_2"] = e1_2
                elif "a1" in galaxy.colnames and "b1" in galaxy.colnames:
                    axis_ratio_1 = axis_ratio(a=galaxy["a1"], b=galaxy["b1"])
                    ellip_1 = eccentricity(q=axis_ratio_1)
                    e1_1, e1_2 = galaxy_projected_eccentricity(float(ellip_1))
                    galaxy["e1_1"] = e1_1
                    galaxy["e1_2"] = e1_2
                else:
                    raise ValueError(
                        "ellipticity or semi-major and semi-minor axis are missing for"
                        "the second light profile in galaxy_list columns"
                    )
            if galaxy["angular_size0"] == -1 or galaxy["angular_size1"] == -1:
                if "a0" in galaxy.colnames and "b0" in galaxy.colnames:
                    galaxy["angular_size0"] = average_angular_size(
                        a=galaxy["a0"], b=galaxy["b0"]
                    )
                else:
                    raise ValueError(
                        "semi-major and semi-minor axis are missing for the first light"
                        "profile in galaxy_list columns"
                    )
                if "a1" in galaxy.colnames and "b1" in galaxy.colnames:
                    galaxy["angular_size1"] = average_angular_size(
                        a=galaxy["a1"], b=galaxy["b1"]
                    )
                else:
                    raise ValueError(
                        "semi-major and semi-minor axis are missing for the second"
                        "light profile in galaxy_list columns"
                    )
            if galaxy["n_sersic_0"] == -1 or galaxy["n_sersic_1"] == -1:
                galaxy["n_sersic_0"] = 1
                galaxy["n_sersic_1"] = 4
        else:
            raise ValueError(
                "Provided number of light profiles is not supported. It should be"
                "either 'single or 'double' "
            )
        return galaxy


def galaxy_projected_eccentricity(ellipticity, rotation_angle=None):
    """Projected eccentricity of elliptical galaxies as a function of other deflector
    parameters.

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param rotation_angle: rotation angle of the major axis of elliptical galaxy in
        radian. The reference of this rotation angle is +Ra axis i.e towards the East
        direction and it goes from East to North. If it is not provided, it will be
        drawn randomly.
    :return: e1, e2 eccentricity components
    """
    if rotation_angle is None:
        phi = np.random.uniform(0, np.pi)
    else:
        phi = rotation_angle
    e = param_util.epsilon2e(ellipticity)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2


def convert_to_slsim_convention(
    galaxy_catalog, light_profile, input_catalog_type="skypy"
):
    """This function converts scotch/catalog to slsim conventions. In slsim, sersic
    index are either n_sersic or (n_sersic_0 and n_sersic_1). Ellipticity are either
    ellipticity or (ellipticity0 and ellipticity1). These kewords can be read by
    Galaxies class. This function is written to convert scotch catalog to slsim
    convension and to change unit of angular size in skypy source catalog to arcsec.

    :param galaxy_catalog: galaxy catalog in other conventions.
    :param light_profile: keyword for number of sersic profile to use in source light
        model. accepted kewords: "single_sersic", "double_sersic".
    :return: galaxy catalog in slsim convension.
    """
    column_names = galaxy_catalog.colnames
    for col_name in column_names:
        if "_host" in col_name:
            # Remove '_host' from the column name
            new_col_name = col_name.replace("_host", "")
            # Rename the column
            galaxy_catalog.rename_column(col_name, new_col_name)
    if light_profile == "double_sersic":
        if "n0" in column_names or "n1" in column_names:
            galaxy_catalog.rename_column("n0", "n_sersic_0")
            galaxy_catalog.rename_column("n1", "n_sersic_1")
        if "e0" in column_names or "e1" in column_names:
            galaxy_catalog.rename_column("e0", "ellipticity0")
            galaxy_catalog.rename_column("e1", "ellipticity1")
    if light_profile == "single_sersic":
        if "e" in column_names:
            galaxy_catalog.rename_column("e", "ellipticity")
    if input_catalog_type == "scotch":
        galaxy_catalog["a_rot"] = np.deg2rad(galaxy_catalog["a_rot"])
    if input_catalog_type == "skypy":
        galaxy_catalog["angular_size"] = (
            galaxy_catalog["angular_size"] / constants.arcsec
        )
    return galaxy_catalog
