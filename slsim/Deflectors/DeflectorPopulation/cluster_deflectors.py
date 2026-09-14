import numpy as np
from numpy import random
from slsim.Lenses.selection import object_cut
from slsim.Deflectors.MassLightConnection.richness2mass import mass_richness_relation
from slsim.Halos.halo_population import gene_e_ang_halo
from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase
from slsim.Deflectors.deflector_group import DeflectorGroup
from slsim.Deflectors.deflector_util import (
    deflector_dict_from_table,
    set_colossus_cosmo,
)
from lenstronomy.Util.param_util import phi_q2_ellipticity
from astropy import units as u
from astropy.table import hstack
from scipy.spatial.distance import cdist


class ClusterDeflectors:
    """Class describing cluster lens model with a NFW profile for the dark
    matter halo and EPL profile for the subhalos (cluster members). It makes
    use of a group/cluster catalog and a group/cluster member catalog (e.g.
    redMaPPer).

    This class is called by setting deflector_type == "cluster-catalog"
    in LensPop.
    """

    def __init__(
        self,
        cluster_list,
        members_list,
        galaxies,
        kwargs_cut,
        kwargs_mass2light,
        cosmo,
        sky_area,
        halo_mass_type="NFW",
        bcg_light_type="single_sersic",
        subhalo_mass_type="EPL",
        galaxy_light_type="single_sersic",
        gamma_pl=None,
        richness_fn="Abdullah2022",
        kwargs_draw_members=None,
        assign_galaxy_redshift=False,
    ):
        """

        :param cluster_list: list of dictionary with redshift and richness
            (or mass) from a group/cluster catalog.
            Mandatory keys: 'cluster_id' 'z', 'richness' or 'halo_mass'
        :type cluster_list: ~astropy.table.Table
        :param members_list: list of dictionary with positions and magnitudes of
            group/cluster members.
            Mandatory keys: 'cluster_id', 'ra', 'dec', 'mag_{band}'
        :type members_list: ~astropy.table.Table
        :param galaxies: SLSim galaxies to be assigned as deflectors to each member.
        :type galaxies: ~slsim.Sources.SourcePopulation.galaxies.Galaxies() class instance
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation matching inputs to deflector_util.mass2light()
        :type kwargs_mass2light: dict
        :param cosmo: astropy cosmology instance
        :type cosmo: ~astropy.cosmology
        :param sky_area: Sky area over which galaxy_list is sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param kwargs_draw_members: kwargs for draw_members method
        :type kwargs_draw_members: dict or None
        :param richness_fn: richness-mass relation to assign a mass to each cluster
        :type richness_fn: str
        :param assign_galaxy_redshift: if True, assign the redshift of the
            galaxy to the member galaxy instead of the cluster redshift
        :type assign_galaxy_redshift: bool
        """
        self._test_consistency_clusters(cluster_list)
        self._galaxies = galaxies
        self._subhalo_mass_type = subhalo_mass_type
        self._galaxy_light_type = galaxy_light_type
        self._cosmo = cosmo
        self.sky_area = sky_area
        self._richness_fn = richness_fn
        # make sure kwargs_mass2light has m_star_v_disp_scaling=True set
        self._kwargs_mass2light = kwargs_mass2light
        self._halo_deflector = DeflectorsBase(
            deflector_table=cluster_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
            mass_type=halo_mass_type,
            light_type=bcg_light_type,
            kwargs_mass2light=kwargs_mass2light,
        )
        if kwargs_draw_members is None:
            kwargs_draw_members = {}
        self._kwargs_draw_members = kwargs_draw_members
        self.set_cosmo()
        # assign a similar SLSim galaxy to each member
        members_list, use_radec = self._preprocess_members(cluster_list, members_list)
        self._use_radec = use_radec
        members_list = self.assign_similar_galaxy(
            members_list=members_list,
            galaxy_list=galaxies._objects_select,
            cosmo=self._cosmo,
            assign_galaxy_redshift=assign_galaxy_redshift,
        )

        self._members_select = object_cut(members_list, **kwargs_cut)
        self._cluster_select = cluster_list[
            np.isin(cluster_list["cluster_id"], self._members_select["cluster_id"])
        ]

        self._num_select = len(self._cluster_select)

    def deflector_number(self):
        """

        :return: number of deflectors
        """
        number = self._num_select
        return number

    def draw_deflector(self, z_max=None, z_min=None, deflector_index=None):
        """

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param deflector_index: index of deflector to pic (if provided)
        :return: DeflectorGroup() class instance of the full cluster
        """
        # TODO: add redshift pre-selection z_min, z_max
        if deflector_index is None:
            deflector_index = random.randint(0, self._num_select - 1)
        deflector = self.draw_cluster(deflector_index)
        z, center_x, center_y, kwargs_mass, kwargs_light = deflector_dict_from_table(
            table=deflector,
            mass_type=self._halo_deflector.mass_type,
            extended_source_type=None,
            cosmo=self._cosmo,
            **self._kwargs_mass2light,
        )

        kwargs_mass_list = [kwargs_mass]
        kwargs_light_list = [kwargs_light]
        center_x_deflector_list = [center_x]
        center_y_deflector_list = [center_y]

        members = self._draw_members(
            deflector["cluster_id"],
            use_radec=self._use_radec,
            **self._kwargs_draw_members,
        )

        for suhalo in members:
            _, center_x_i, center_y_i, kwargs_mass_i, kwargs_light_i = (
                deflector_dict_from_table(
                    table=suhalo,
                    mass_type=self._subhalo_mass_type,
                    extended_source_type=self._galaxy_light_type,
                    m_star_v_disp_scaling=True,
                    **self._kwargs_mass2light,
                )
            )
            kwargs_mass_list.append(kwargs_mass_i)
            kwargs_light_list.append(kwargs_light_i)
            center_x_deflector_list.append(center_x_i)
            center_y_deflector_list.append(center_y_i)
        deflector_group = DeflectorGroup(
            z=z,
            kwargs_mass_list=kwargs_mass_list,
            kwargs_light_list=kwargs_light_list,
            center_x_deflector_list=center_x_deflector_list,
            center_y_deflector_list=center_y_deflector_list,
            center_x=0,
            center_y=0,
        )

        return deflector_group

    def draw_cluster(self, index):
        """
        :param index: index of cluster in catalog
        :type index: int

        :return: dictionary of NFW parameters for the cluster halo
        """
        cluster = self._cluster_select[index]
        halo_columns = cluster.colnames
        cluster_dict = dict(cluster)
        if "halo_mass" not in halo_columns:
            if "richness" not in halo_columns:
                raise ValueError(
                    "Either 'halo_mass' or 'richness' needs to be in cluster_catalog"
                )
            halo_mass = mass_richness_relation(
                cluster["richness"], relation=self._richness_fn
            )
            cluster_dict["halo_mass"] = halo_mass

        if "e1_mass" not in halo_columns or "e2_mass" not in halo_columns:
            e, phi = gene_e_ang_halo(np.array([cluster_dict["halo_mass"]]))
            e1, e2 = phi_q2_ellipticity(np.deg2rad(phi[0]), 1 - e[0])
            cluster_dict["e1_mass"] = e1
            cluster_dict["e2_mass"] = e2
        return cluster_dict

    def _draw_members(
        self, cluster_id, center_scatter=0.0, max_dist=80, bcg_band="r", use_radec=False
    ):
        """Draw cluster members relative to (0,0) as cluster center.

        :param cluster_id: identifier of the cluster
        :type cluster_id: int
        :param center_scatter: scatter in center of the BCG in arcsec
        :type center_scatter: float
        :param max_dist: maximum distance from the BCG in arcsec
        :type max_dist: float
        bcg_band: band to use to identify the BCG
        :type bcg_band: str
        :param use_radec: if True, reads 'ra' and 'dec' from members, otherwise 'center_x', 'center_y'
        :return: astropy table with EPL+Sersic parameters of each member
        """
        members = self._members_select[cluster_id == self._members_select["cluster_id"]]

        bcg_id = np.argmin(members[f"mag_{bcg_band}"])
        if use_radec:

            bcg_ra, bcg_dec = members["ra"][bcg_id], members["dec"][bcg_id]
            center_ra, center_dec = (
                np.random.normal(
                    bcg_ra, center_scatter / 3600 / np.cos(bcg_dec / 180 * np.pi)
                ),
                np.random.normal(bcg_dec, center_scatter / 3600),
            )
            center_x = (
                (members["ra"] - center_ra) * 3600 * np.cos(center_dec / 180 * np.pi)
            )
            center_y = (members["dec"] - center_dec) * 3600
        else:
            bcg_x = members["center_x"][bcg_id]
            bcg_y = members["center_y"][bcg_id]
            bcg_x = np.random.normal(bcg_x, center_scatter)
            bcg_y = np.random.normal(bcg_y, center_scatter)
            center_x = members["center_x"] - bcg_x
            center_y = members["center_y"] - bcg_y
        members["center_x"] = center_x
        members["center_y"] = center_y
        center_dist = np.sqrt(center_x**2 + center_y**2)
        members = members[center_dist < max_dist]
        return members

    @staticmethod
    def assign_similar_galaxy(
        members_list,
        galaxy_list,
        cosmo=None,
        bands=("g", "r", "i", "z", "Y"),
        max_gals=10000,
        assign_galaxy_redshift=False,
    ):
        """Assigns a similar galaxy to each member of a group/cluster member
        catalog by comparing their magnitudes and redshifts.

        :param members_list: astropy table with columns 'mag_{band}',
            'z'
        :type members_list: astropy.table.Table
        :param galaxy_list: astropy table with columns 'mag_{band}', 'z'
        :type galaxy_list: astropy.table.Table
        :param cosmo: astropy.cosmology instance
        :type cosmo: astropy.cosmology
        :param bands: list of bands to compare
        :type bands: list
        :param max_gals: maximum number of galaxies to compare to
        :type max_gals: int
        :param assign_galaxy_redshift: if True, assign the redshift of
            the galaxy to the member galaxy instead of the cluster
            redshift
        :type assign_galaxy_redshift: bool
        :return: astropy table with the same number of rows as
            members_list and columns from both members_list and
            galaxy_list
        :rtype: astropy.table.Table
        """
        # TODO: this routine needs to be revised to match bolometric brightness or even stellar masses instead
        #  of apparent magnitudes, as it might lead to few galaxies being matched
        # shuffle galaxy list and select a subset
        if len(galaxy_list) > max_gals:
            indices = np.random.choice(len(galaxy_list), max_gals, replace=False)
            galaxy_list = galaxy_list[indices]

        mag_cols = [f"mag_{b}" for b in bands if f"mag_{b}" in members_list.columns]
        if not mag_cols:
            raise ValueError("No magnitude columns found in members_list")
        mag_members = [members_list[mag] for mag in mag_cols]
        mag_galaxies = [galaxy_list[mag] for mag in mag_cols]
        dist_mod_members = -5 * np.log10(
            cosmo.luminosity_distance(members_list["z"]) / (10 * u.pc)
        )
        dist_mod_galaxies = -5 * np.log10(
            cosmo.luminosity_distance(galaxy_list["z"]) / (10 * u.pc)
        )
        distance = cdist(
            np.stack([*mag_members, dist_mod_members], axis=1),
            np.stack([*mag_galaxies, dist_mod_galaxies], axis=1),
            metric="euclidean",
        )
        nearest_neighbors_indices = distance.argmin(axis=1)
        similar_galaxies = galaxy_list[nearest_neighbors_indices]

        if assign_galaxy_redshift:
            # Use galaxy redshift instead of member redshift
            include_cols_members = [
                col
                for col in members_list.columns
                if col not in mag_cols + ["z"]  # Exclude both mags AND redshift
            ]
            include_cols_galaxies = [
                col
                for col in galaxy_list.columns  # Keep ALL galaxy columns including 'z'
            ]
        else:
            # Original behavior - use member redshift
            include_cols_members = [
                col for col in members_list.columns if col not in mag_cols
            ]
            include_cols_galaxies = [
                col for col in galaxy_list.columns if col not in ["z"]
            ]

        return hstack(
            [
                members_list[include_cols_members],
                similar_galaxies[include_cols_galaxies],
            ]
        )

    @staticmethod
    def _test_consistency_clusters(cluster_list):
        column_names = cluster_list.columns

        if "cluster_id" not in column_names:
            raise ValueError("cluster_id is mandatory in cluster catalog")
        if "z" not in column_names:
            raise ValueError("redshift is mandatory in cluster catalog")
        if "halo_mass" not in column_names and "richness" not in column_names:
            raise ValueError("richness or halo_mass is mandatory in cluster catalog")

    @staticmethod
    def _preprocess_members(cluster_list, members_list):
        """Make sure members have redshift entries.

        :param cluster_list: cluster list
        :param members_list: member list
        :return: updated members list, use_radec
        """
        n_clusters = len(cluster_list)
        n_members = len(members_list)
        column_names = members_list.columns
        if "z" not in column_names:
            members_list["z"] = -np.ones(n_members)
            # assign the redshift of the cluster to its members
            for i in range(n_clusters):
                z = cluster_list["z"][i]
                members_list["z"][
                    members_list["cluster_id"] == cluster_list["cluster_id"][i]
                ] = z
        if "center_x" not in column_names or "center_y" not in column_names:
            members_list["center_x"] = -np.ones(n_members)
            members_list["center_y"] = -np.ones(n_members)
            use_radec = True
            if "ra" not in column_names or "dec" not in column_names:
                raise ValueError(
                    "either 'center_x', 'center_y' or 'ra', 'dec' have to be provided for members."
                )
        else:
            use_radec = False
        return members_list, use_radec

    def set_cosmo(self):
        """Set the cosmology in colossus to match the astropy.cosmology
        instance."""
        set_colossus_cosmo(cosmo=self._cosmo)
