import numpy as np
import numpy.random as random
from slsim.Util import param_util
from slsim.Lenses.selection import object_cut
from slsim.Deflectors.MassLightConnection.richness2mass import mass_richness_relation
from slsim.Halos.halo_population import gene_e_ang_halo, concent_m_w_scatter
from colossus.cosmology import cosmology as colossus_cosmo
from slsim.Deflectors.MassLightConnection.velocity_dispersion import (
    vel_disp_abundance_matching,
)
from slsim.Deflectors.DeflectorPopulation.elliptical_lens_galaxies import (
    elliptical_projected_eccentricity,
)
from slsim.Deflectors.MassLightConnection.velocity_dispersion import vel_disp_nfw
from slsim.Deflectors.DeflectorPopulation.deflectors_base import DeflectorsBase
from slsim.Deflectors.deflector import Deflector
from lenstronomy.Util.param_util import phi_q2_ellipticity
from astropy import units as u
from astropy.table import hstack
from scipy.spatial.distance import cdist


class ClusterDeflectors(DeflectorsBase):
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
        galaxy_list,
        kwargs_cut,
        kwargs_mass2light,
        cosmo,
        sky_area,
        catalog_type="skypy",
        richness_fn="Abdullah2022",
        kwargs_draw_members=None,
        assign_galaxy_redshift=False,
        cored_profile=False,
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
        :param galaxy_list: list of dictionary with lens parameters of
            SLSim galaxies to be assigned as deflectors to each member.
        :type galaxy_list: ~astropy.table.Table
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :type kwargs_mass2light: dict
        :param cosmo: astropy.cosmology instance
        :type cosmo: ~astropy.cosmology
        :param sky_area: Sky area over which galaxy_list is sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param richness_fn: richness-mass relation to assign a mass to each cluster
        :type richness_fn: str
        :param kwargs_draw_members: kwargs for draw_members method
        :type kwargs_draw_members: dict or None
        :param catalog_type: type of input catalog (skypy or cosmoDC2)
        :type catalog_type: str
        :param assign_galaxy_redshift: if True, assign the redshift of the
            galaxy to the member galaxy instead of the cluster redshift
        :type assign_galaxy_redshift: bool
        :param cored_profile: flag for adding cored density profile
        :type cored_profile: boolean
        """
        galaxy_list = param_util.catalog_with_angular_size_in_arcsec(
            galaxy_catalog=galaxy_list, input_catalog_type=catalog_type
        )
        super().__init__(
            deflector_table=cluster_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        self.deflector_profile = "NFW_CLUSTER"
        self.cored_profile = cored_profile
        self.richness_fn = richness_fn
        if kwargs_draw_members is None:
            kwargs_draw_members = {}
        self.kwargs_draw_members = kwargs_draw_members
        self.set_cosmo()

        cluster_list = self.preprocess_clusters(cluster_list)
        members_list = self.preprocess_members(
            cluster_list,
            members_list,
            galaxy_list,
            assign_galaxy_redshift=assign_galaxy_redshift,
        )

        self._f_vel_disp = vel_disp_abundance_matching(
            galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
        )

        self._members_select = object_cut(members_list, **kwargs_cut)
        self._cluster_select = cluster_list[
            np.isin(cluster_list["cluster_id"], self._members_select["cluster_id"])
        ]

        self._members_select["vel_disp"] = self._f_vel_disp(
            np.log10(self._members_select["stellar_mass"])
        )

        self._kwargs_mass2light = kwargs_mass2light

        self._num_select = len(self._cluster_select)

        self._cosmo = cosmo

        # TODO: random reshuffle of matched list

    def deflector_number(self):
        """

        :return: number of deflectors
        """
        number = self._num_select
        return number

    def draw_deflector(self, index=None):
        """
        :param index: index of deflector, if not provided, draw randomly from all deflectors
        :type index: int or None
        :param cored: Boolean flag for cored density profile
        :type cored: True for cored, False for cuspy profile
        :return: dictionary of complete parameterization of deflector
        """
        index = random.randint(0, self._num_select - 1)
        deflector = self.draw_cluster(index)
        members = self.draw_members(deflector["cluster_id"], **self.kwargs_draw_members)
        deflector["subhalos"] = members
        deflector["cored_profile"] = self.cored_profile
        deflector_class = Deflector(deflector_type=self.deflector_profile, **deflector)
        return deflector_class

    def get_deflector(self, cluster_id, cored=False):
        """
        :param cluster_id: identifier of the cluster
        :type cluster_id: int
        :type index: int or None
        :param cored: Boolean flag for cored density profile
        :type cored: True for cored, False for cuspy profile
        :return: dictionary of complete parameterization of deflector for the given cluster_id
        """
        indices = np.where(self._cluster_select["cluster_id"] == cluster_id)[0]

        index = indices[0]  # Take the first match

        # Draw the cluster using the found index
        deflector = self.draw_cluster(index)

        # Draw the members for this cluster
        members = self.draw_members(deflector["cluster_id"], **self.kwargs_draw_members)
        deflector["subhalos"] = members
        deflector["cored_profile"] = self.cored_profile
        # Create and return the deflector class
        deflector_class = Deflector(deflector_type=self.deflector_profile, **deflector)
        return deflector_class

    def draw_cluster(self, index):
        """
        :param index: index of cluster in catalog
        :type index: int
        :return: dictionary of NFW parameters for the cluster halo
        """
        cluster = self._cluster_select[index]
        if cluster["halo_mass"] == -1:
            cluster["halo_mass"] = mass_richness_relation(
                cluster["richness"], self.richness_fn
            )
        if cluster["concentration"] == -1:
            cluster["concentration"] = concent_m_w_scatter(
                np.array([cluster["halo_mass"]]), cluster["z"], sig=0.33
            )[0]
        if cluster["vel_disp"] == -1:
            cluster["vel_disp"] = vel_disp_nfw(
                cluster["halo_mass"], cluster["concentration"], self.cosmo, cluster["z"]
            )
        if cluster["e1_mass"] == -1 or cluster["e2_mass"] == -1:
            e, phi = gene_e_ang_halo(np.array([cluster["halo_mass"]]))
            e1, e2 = phi_q2_ellipticity(np.deg2rad(phi[0]), 1 - e[0])
            cluster["e1_mass"] = e1
            cluster["e2_mass"] = e2
        return dict(cluster)

    def draw_members(self, cluster_id, center_scatter=0.2, max_dist=80, bcg_band="r"):
        """
        :param cluster_id: identifier of the cluster
        :type cluster_id: int
        :param center_scatter: scatter in center of the BCG in arcsec
        :type center_scatter: float
        :param max_dist: maximum distance from the BCG in arcsec
        :type max_dist: float
        bcg_band: band to use to identify the BCG
        :type bcg_band: str
        :return: astropy table with EPL+Sersic parameters of each member
        """
        members = self._members_select[cluster_id == self._members_select["cluster_id"]]

        members["vel_disp"] = np.where(
            members["vel_disp"] == -1,
            param_util.vel_disp_from_m_star(members["stellar_mass"]),
            members["vel_disp"],
        )

        for i in range(len(members)):
            if members[i]["e1_light"] == -1 or members[i]["e2_light"] == -1:
                e1_light, e2_light, e1_mass, e2_mass = (
                    elliptical_projected_eccentricity(
                        **members[i], **self._kwargs_mass2light
                    )
                )
                members[i]["e1_light"] = e1_light
                members[i]["e2_light"] = e2_light
                members[i]["e1_mass"] = e1_mass
                members[i]["e2_mass"] = e2_mass
        members["n_sersic"] = np.where(
            members["n_sersic"] == -1, 4, members["n_sersic"]
        )
        bcg_id = np.argmin(members[f"mag_{bcg_band}"])
        bcg_ra, bcg_dec = members["ra"][bcg_id], members["dec"][bcg_id]
        center_ra, center_dec = (
            np.random.normal(bcg_ra, center_scatter / 3600),
            np.random.normal(bcg_dec, center_scatter / 3600),
        )
        center_x = (members["ra"] - center_ra) * 3600 * np.cos(center_dec / 180 * np.pi)
        center_y = (members["dec"] - center_dec) * 3600
        members["center_x"] = np.where(
            members["center_x"] == -1, center_x, members["center_x"]
        )
        members["center_y"] = np.where(
            members["center_y"] == -1, center_y, members["center_y"]
        )
        center_dist = np.sqrt(members["center_x"] ** 2 + members["center_y"] ** 2)
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
    def preprocess_clusters(cluster_list):
        n_clusters = len(cluster_list)
        column_names = cluster_list.columns

        if "cluster_id" not in column_names:
            raise ValueError("cluster_id is mandatory in cluster catalog")
        if "z" not in column_names:
            raise ValueError("redshift is mandatory in cluster catalog")
        if "halo_mass" not in column_names:
            if "richness" not in column_names:
                raise ValueError(
                    "richness or halo_mass is mandatory in cluster catalog"
                )
            cluster_list["halo_mass"] = -np.ones(n_clusters)
        if "concentration" not in column_names:
            cluster_list["concentration"] = -np.ones(n_clusters)
        if "vel_disp" not in column_names:
            cluster_list["vel_disp"] = -np.ones(n_clusters)
        if "e1_mass" not in column_names or "e2_mass" not in column_names:
            cluster_list["e1_mass"] = -np.ones(n_clusters)
            cluster_list["e2_mass"] = -np.ones(n_clusters)
        return cluster_list

    def preprocess_members(
        self, cluster_list, members_list, galaxy_list, assign_galaxy_redshift=False
    ):
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
        # use center_x and center_y if available, otherwise use ra and dec
        if "center_x" not in column_names or "center_y" not in column_names:
            members_list["center_x"] = -np.ones(n_members)
            members_list["center_y"] = -np.ones(n_members)
            if "ra" not in column_names or "dec" not in column_names:
                raise ValueError(
                    "ra and dec or center_x and center_y "
                    "are mandatory in members catalog"
                )
        else:
            if "ra" not in column_names or "dec" not in column_names:
                members_list["ra"] = -np.ones(n_members)
                members_list["dec"] = -np.ones(n_members)
        # assign a similar SLSim galaxy to each member
        members_list = self.assign_similar_galaxy(
            members_list,
            galaxy_list,
            cosmo=self.cosmo,
            assign_galaxy_redshift=assign_galaxy_redshift,
        )
        # update column names
        column_names = members_list.colnames
        if "vel_disp" not in column_names:
            members_list["vel_disp"] = -np.ones(n_members)
        if "e1_light" not in column_names or "e2_light" not in column_names:
            members_list["e1_light"] = -np.ones(n_members)
            members_list["e2_light"] = -np.ones(n_members)
        if "e1_mass" not in column_names or "e2_mass" not in column_names:
            members_list["e1_mass"] = -np.ones(n_members)
            members_list["e2_mass"] = -np.ones(n_members)
        if "n_sersic" not in column_names:
            members_list["n_sersic"] = -np.ones(n_members)
        if "gamma_pl" not in column_names:
            members_list["gamma_pl"] = np.ones(n_members) * 2
        return members_list

    def set_cosmo(self):
        """Set the cosmology in colossus to match the astropy.cosmology
        instance."""
        params = dict(
            flat=(self.cosmo.Ok0 == 0.0),
            H0=self.cosmo.H0.value,
            Om0=self.cosmo.Om0,
            Ode0=self.cosmo.Ode0,
            Ob0=(
                self.cosmo.Ob0
                if (self.cosmo.Ob0 is not None) and (self.cosmo.Ob0 != 0)
                else 0.04897
            ),
            Tcmb0=self.cosmo.Tcmb0.value if self.cosmo.Tcmb0.value > 0 else 2.7255,
            Neff=self.cosmo.Neff,
            sigma8=0.8102,
            ns=0.9660499,
        )
        colossus_cosmo.setCosmology(cosmo_name="halo_cosmo", **params)
