import numpy as np
import numpy.random as random
from slsim.selection import object_cut
from slsim.Util import param_util
from slsim.Sources.source_pop_base import SourcePopBase
from astropy.table import Column, vstack
from slsim.Util.param_util import (
    average_angular_size,
    axis_ratio,
    eccentricity,
    downsample_galaxies,
    galaxy_size_redshift_evolution
)
from astropy import units as u
from slsim.Sources.source import Source
import os


# TODO: Use type to determine galaxy_list type
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
        downsample_to_dc2=False,
    ):
        """

        :param galaxy_list: An astropy table with galaxy parameters.
        :type galaxy_list: astropy Table object.
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
        :param downsample_to_dc2: Boolean. If True, downsamples the given galaxy
         population at redshift greater than 1.5 to DC2 galaxy population.
        """
        super(Galaxies, self).__init__(cosmo=cosmo, sky_area=sky_area)
        self.source_type = "extended"
        self.light_profile = light_profile
        if downsample_to_dc2 is True:
            samp1, samp2, samp3, samp4, samp5, samp6 = down_sample_to_dc2(
                galaxy_pop=galaxy_list, sky_area=sky_area
            )
            samp_low = galaxy_list[galaxy_list["z"] <= 2]
            galaxy_list = vstack([samp_low, samp1, samp2, samp3, samp4, samp5, samp6])
            """slsim_sample_3_35, slsim_sample_35_4, slsim_sample_4_45,
            slsim_sample_45_5])"""
        self.n = len(galaxy_list)
        # add missing keywords in astropy.Table object
        if list_type == "astropy_table":
            galaxy_list = convert_to_slsim_convention(
                galaxy_catalog=galaxy_list,
                light_profile=self.light_profile,
                input_catalog_type=catalog_type,
                cosmo=cosmo
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
        self._galaxy_select = object_cut(galaxy_list, list_type=list_type, **kwargs_cut)
        self._num_select = len(self._galaxy_select)
        self.list_type = list_type

    @property
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        return self.n

    @property
    def source_number_selected(self):
        """Number of sources selected (within given area on the sky)

        :return: number of sources passing the selection criteria
        """
        return self._num_select

    def draw_source(self, z_max=None):
        """Choose source at random. :param z_max: maximum redshift for source
        to be drawn.

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :return: dictionary of source
        """
        if z_max is not None:
            filtered_galaxies = self._galaxy_select[self._galaxy_select["z"] < z_max]
            if len(filtered_galaxies) == 0:
                return None
            else:
                index = random.randint(0, len(filtered_galaxies) - 1)
                galaxy = filtered_galaxies[index]
        else:
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
        source_class = Source(
            source_dict=galaxy,
            variability_model=self.variability_model,
            kwargs_variability=self.kwargs_variability,
            sn_type=self.sn_type,
            sn_absolute_mag_band=self.sn_absolute_mag_band,
            sn_absolute_zpsys=self.sn_absolute_zpsys,
            cosmo=self._cosmo,
            lightcurve_time=self.lightcurve_time,
            sn_modeldir=self.sn_modeldir,
            agn_driving_variability_model=self.agn_driving_variability_model,
            agn_driving_kwargs_variability=self.agn_driving_kwargs_variability,
            source_type=self.source_type,
            light_profile=self.light_profile,
        )
        return source_class


def galaxy_projected_eccentricity(ellipticity, rotation_angle=None):
    """Projected eccentricity of elliptical galaxies as a function of other
    deflector parameters.

    :param ellipticity: eccentricity amplitude
    :type ellipticity: float [0,1)
    :param rotation_angle: rotation angle of the major axis of
        elliptical galaxy in radian. The reference of this rotation
        angle is +Ra axis i.e towards the East direction and it goes
        from East to North. If it is not provided, it will be drawn
        randomly.
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
    galaxy_catalog, light_profile, input_catalog_type="skypy", cosmo=None
):
    """This function converts scotch/catalog to slsim conventions. In slsim,
    sersic index are either n_sersic or (n_sersic_0 and n_sersic_1).
    Ellipticity are either ellipticity or (ellipticity0 and ellipticity1).
    These kewords can be read by Galaxies class. This function is written to
    convert scotch catalog to slsim convension and to change unit of angular
    size in skypy source catalog to arcsec.

    :param galaxy_catalog: An astropy table of galaxy catalog in other
        conventions.
    :type galaxy_catalog: astropy Table object.
    :param light_profile: keyword for number of sersic profile to use in
        source light model. accepted kewords: "single_sersic",
        "double_sersic".
    :return: galaxy catalog in slsim convension.
    """
    galaxy_catalog = galaxy_catalog.copy()
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
        # compute the rescaled physical size. The resulted value is devided by 2.5 to
        #  match the best-fit model given in https://iopscience.iop.org/article/10.1088/0067-0049/219/2/15/pdf
        rescaled_physical_size=galaxy_catalog["physical_size"]*galaxy_size_redshift_evolution(
            galaxy_catalog["z"])/2.5
        # compute the rescaled angular size
        rescaled_angular_size = (rescaled_physical_size)/cosmo.angular_diameter_distance(
            galaxy_catalog["z"]).to(u.kpc)
        galaxy_catalog["physical_size"] = rescaled_physical_size
        galaxy_catalog["angular_size"] = (rescaled_angular_size*u.rad).to(u.arcsec)
    return galaxy_catalog


def down_sample_to_dc2(galaxy_pop, sky_area):
    """Downsamples given galaxy pop above redshift 1.5 to DC2 galaxy
    population.

    :param galaxy_pop: Astropy table of galaxy population.
    :param sky_area: Sky area over which galaxies are sampled. Must be in units of
     solid angle and it should be astropy unit object.
    :param cosmo: astropy.cosmology instance
    :return: Astropy tables of downsampled galaxy population in different bins.
     Redshift bins for returned populations are: (2-2.5), (2.5-3), (3-3.5),
     (3.5-4), (4-4.5), (4.5-5)
    """
    path = os.path.dirname(__file__)
    new_path = path[: path.rfind("slsim/")]
    module_path = os.path.dirname(new_path)
    # path1 = os.path.join(
    #    module_path, "data/DC2_data/dc2_galaxy_count_1.5_2.npy"
    # )
    path2 = os.path.join(module_path, "data/DC2_data/dc2_galaxy_count_2_2.5.npy")
    path3 = os.path.join(module_path, "data/DC2_data/dc2_galaxy_count_2.5_3.npy")
    # DC2 galaxy counts in 3 different redshift bins: (2-2.5), (2.5-3). Beyond 3
    # , we use the same count as 3rd bin because DC2 only reach up to redshift 3.
    # dN1 = np.load(path1)
    dN2 = int(sky_area.value) * np.load(path2)
    dN3 = int(sky_area.value) * np.load(path3)

    # M_min1=21.531229
    # M_max1=29.999994
    # dM1=0.2920263882341056
    M_min2 = 22.084414
    M_max2 = 29.999998
    dM2 = 0.2729511918692753
    M_min3 = 22.654068
    M_max3 = 29.999996
    dM3 = 0.25330786869443694
    # slsim_sample_15_2=downsample_galaxies(galaxy_pop, dN1, dM1, M_min1,
    #                                        M_max1, 1.5, 2)
    slsim_sample_2_25 = downsample_galaxies(
        galaxy_pop, dN2, dM2, M_min2, M_max2, 2, 2.5
    )
    slsim_sample_25_3 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 2.5, 3
    )
    slsim_sample_3_35 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 3, 3.5
    )
    slsim_sample_35_4 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 3.5, 4
    )
    slsim_sample_4_45 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 4, 4.5
    )
    slsim_sample_45_5 = downsample_galaxies(
        galaxy_pop, dN3, dM3, M_min3, M_max3, 4.5, 5
    )
    return (
        slsim_sample_2_25,
        slsim_sample_25_3,
        slsim_sample_3_35,
        slsim_sample_35_4,
        slsim_sample_4_45,
        slsim_sample_45_5,
    )
