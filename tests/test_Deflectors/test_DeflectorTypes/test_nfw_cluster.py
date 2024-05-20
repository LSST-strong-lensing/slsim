from slsim.Deflectors.DeflectorTypes.nfw_cluster import NFWCluster
from slsim.Deflectors.deflector import Deflector
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import os
import numpy.testing as npt


class TestNFWCluster(object):
    """
    required quantities in dictionary:
    - 'halo_mass': halo mass in physical M_sol
    - 'concentration': halo concentration
    - 'e1_mass': eccentricity of NFW profile
    - 'e2_mass': eccentricity of NFW profile
    - 'z': redshift of deflector
    - subhalos_list: list of subhalos, each one is a deflector instance
    """

    def setup_method(self):
        path = os.path.dirname(__file__)
        module_path = os.path.dirname(os.path.dirname(path))
        # a table with the dictionary for a single dark matter halo
        self.halo_dict = Table.read(
            os.path.join(module_path, "TestData/halo_NFW.fits"), format="fits"
        )
        # a table with the dictionary for 10 EPL+Sersic subhalos
        subhalos_table = Table.read(
            os.path.join(module_path, "TestData/subhalos_table.fits"), format="fits"
        )
        subhalos_list = [
            Deflector(deflector_type="EPL", deflector_dict=subhalo)
            for subhalo in subhalos_table
        ]
        self.nfw_cluster = NFWCluster(
            deflector_dict=self.halo_dict, subhalos_list=subhalos_list
        )

    def test_redshift(self):
        z = self.nfw_cluster.redshift
        assert self.halo_dict["z"] == z

    def test_halo_properties(self):
        m_halo, c = self.nfw_cluster.halo_properties
        assert m_halo == self.halo_dict["halo_mass"]
        assert c == self.halo_dict["concentration"]

    def test_velocity_dispersion(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        vel_disp = self.nfw_cluster.velocity_dispersion(cosmo=cosmo)
        npt.assert_almost_equal(vel_disp, 1200, decimal=-1)

    def test_light_model_lenstronomy(self):
        lens_light_model_list, kwargs_lens_light = (
            self.nfw_cluster.light_model_lenstronomy(band="g")
        )
        # one for each subhalo
        assert len(lens_light_model_list) == 10
        assert len(kwargs_lens_light) == 10
