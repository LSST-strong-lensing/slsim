import numpy as np
import numpy.random as random
from slsim.selection import object_cut
from slsim.Deflectors.richness2mass import mass_richness_simet2017
from slsim.Deflectors.halo_population import gene_e_ang_halo, concent_m_w_scatter
from slsim.Deflectors.velocity_dispersion import vel_disp_abundance_matching
from slsim.Deflectors.elliptical_lens_galaxies import elliptical_projected_eccentricity, vel_disp_from_m_star
from slsim.Deflectors.deflectors_base import DeflectorsBase
from lenstronomy.Util.param_util import phi_q2_ellipticity
from astropy import units as u
from astropy.table import hstack
from scipy.spatial.distance import cdist


class ClusterCatalogLens(DeflectorsBase):
    """Class describing cluster lens model with a NFW profile for the dark matter halo
    and EPL profile for the subhalos (cluster members). It makes use of a group/cluster
    catalog and a group/cluster member catalog (e.g. redMaPPer).

    This class is called by setting deflector_type == "cluster-catalog" in LensPop.
    """

    def __init__(
        self, cluster_list, members_list, galaxy_list, kwargs_cut, kwargs_mass2light, cosmo, sky_area
    ):
        """

        :param cluster_list: list of dictionary with lens parameters of
            elliptical dark matter halos from a group/cluster catalog.
            Mandatory keys: 'cluster_id' 'z', 'richness'
        :param members_list: list of dictionary with lens parameters of
            elliptical galaxies from a group/cluster member catalog.
            Mandatory keys: 'cluster_id', 'member_id', 'RA', 'DEC',
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param kwargs_mass2light: mass-to-light relation
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        ## MEMO: DeflectorsBase's inputs are deflector_table, kwargs_cut, cosmo, sky_area
        """
        super().__init__(
            deflector_table=cluster_list,
            kwargs_cut=kwargs_cut,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        
        # cluster
        n_clusters = len(cluster_list)
        cluster_column_names = cluster_list.columns
        if "richness" not in cluster_column_names:
            raise ValueError("richness is mandatory in cluster catalog")
        if "z" not in cluster_column_names:
            raise ValueError("redshift is mandatory in cluster catalog")
        if "halo_mass" not in cluster_column_names:
            cluster_list["halo_mass"] = -np.ones(n_clusters)
        if "concentration" not in cluster_column_names:
            cluster_list["concentration"] = -np.ones(n_clusters)
        if "e1_mass" not in cluster_column_names or "e2_mass" not in cluster_column_names:
            cluster_list["e1_mass"] = -np.ones(n_clusters)
            cluster_list["e2_mass"] = -np.ones(n_clusters)
        
        # members
        if "z" not in members_list.columns:
            # assign the redshift of the cluster to its members
            for i in range(n_clusters):
                z = cluster_list["z"][i]
                members_list["z"][members_list["cluster_id"] == cluster_list["cluster_id"][i]] = z
        # assign a similar SLSim galaxy to each member
        members_list = self.assign_similar_galaxy(members_list, galaxy_list, cosmo=cosmo)
        n_members = len(members_list)
        members_column_names = galaxy_list.colnames
        if "vel_disp" not in members_column_names:
            galaxy_list["vel_disp"] = -np.ones(n_members)
        if "e1_light" not in members_column_names or "e2_light" not in members_column_names:
            galaxy_list["e1_light"] = -np.ones(n_members)
            galaxy_list["e2_light"] = -np.ones(n_members)
        if "e1_mass" not in members_column_names or "e2_mass" not in members_column_names:
            galaxy_list["e1_mass"] = -np.ones(n_members)
            galaxy_list["e2_mass"] = -np.ones(n_members)
        if "n_sersic" not in members_column_names:
            galaxy_list["n_sersic"] = -np.ones(n_members)
        if "gamma_pl" not in members_column_names:
            galaxy_list["gamma_pl"] = np.ones(n_members) * 2

        self._f_vel_disp = vel_disp_abundance_matching(
            galaxy_list, z_max=0.5, sky_area=sky_area, cosmo=cosmo
        )

        self._members_select = object_cut(members_list, **kwargs_cut)
        self._cluster_select = cluster_list[np.isin(cluster_list["cluster_id"],
                                                    self._members_select["cluster_id"])]

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

    def draw_deflector(self):
        """
        :return: dictionary of complete parameterization of deflector
        """
        index = random.randint(0, self._num_select - 1)
        deflector = self.draw_cluster(index)
        members = self.draw_members(deflector["cluster_id"])
        deflector["subhalos"] = members

    def draw_cluster(self, index):
        deflector = self._cluster_select[index]
        if deflector["halo_mass"] == -1:
            deflector["halo_mass"] = mass_richness_simet2017(deflector["richness"])
        if deflector["concentration"] == -1:
            deflector["concentration"] = concent_m_w_scatter(deflector["halo_mass"], deflector["z"], sig=0.33)
        if deflector["e1_mass"] == -1 or deflector["e2_mass"] == -1:
            e, phi = gene_e_ang_halo(deflector["halo_mass"])
            e1, e2 = phi_q2_ellipticity(np.deg2rad(phi), 1 - e)
            deflector["e1_mass"] = e1
            deflector["e2_mass"] = e2
        return deflector

    def draw_members(self, cluster_id):
        # members
        members = self._members_select[cluster_id == self._members_select["cluster_id"]]

        if members["vel_disp"] == -1:
            stellar_mass = members["stellar_mass"]
            vel_disp = vel_disp_from_m_star(stellar_mass)
            members["vel_disp"] = vel_disp
        if members["e1_light"] == -1 or members["e2_light"] == -1:
            for i in range(len(members)):
                e1_light, e2_light, e1_mass, e2_mass = elliptical_projected_eccentricity(
                    **members[i], **self._kwargs_mass2light
                )
                members[i]["e1_light"] = e1_light
                members[i]["e2_light"] = e2_light
                members[i]["e1_mass"] = e1_mass
                members[i]["e2_mass"] = e2_mass
        if members["n_sersic"] == -1:
            members["n_sersic"] = 4  # TODO make a better estimate with scatter
        return members
    
    @staticmethod
    def assign_similar_galaxy(
            members_list, galaxy_list, cosmo=None, bands=("u", "g", "r", "i", "z", "Y")
    ):
        bands = [b for b in bands if f"mag_{b}" in members_list.columns]
        if not bands:
            raise ValueError("No magnitude columns found in members_list")
        mag_members = [members_list[f"mag_{b}"].data for b in bands]
        mag_deflectors = [galaxy_list[f"mag_{b}"].data for b in bands]
        redshifts = galaxy_list["z"].data
        dist_mod_members = (
                -5
                * np.log10(cosmo.luminosity_distance(redshifts) / (10 * u.pc))
                * np.ones(len(members_list))
        )
        dist_mod_deflectors = -5 * np.log10(
            cosmo.luminosity_distance(galaxy_list["z"].data) / (10 * u.pc)
        )
        distance = cdist(
            np.stack([*mag_members, dist_mod_members], axis=1),
            np.stack([*mag_deflectors, dist_mod_deflectors], axis=1),
            metric="euclidean",
        )
        nearest_neighbors_indices = distance.argmin(axis=1)
        similar_deflectors = galaxy_list[nearest_neighbors_indices]
        return hstack(members_list, similar_deflectors)
