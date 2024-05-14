from slsim.Deflectors.DeflectorTypes.nfw_cluster import NFWCluster
from slsim.Deflectors.deflector import Deflector
from astropy.cosmology import FlatLambdaCDM
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
        self.deflector_dict = {
            "halo_mass": 10**15,
            "concentration": 10,
            "e1_mass": 0.1,
            "e2_mass": -0.1,
            "z": 0.5,
        }
        subhalos_list = [
            Deflector(
                deflector_type="EPL",
                deflector_dict={
                    "vel_disp": 200,
                    "e1_mass": 0.1,
                    "e2_mass": -0.1,
                    "angular_size": 0.001,
                    "n_sersic": 1,
                    "e1_light": -0.1,
                    "e2_light": 0.1,
                    "z": 0.5,
                    "mag_g": -18,
                },
            ),
            Deflector(
                deflector_type="NFW_HERNQUIST",
                deflector_dict={
                    "halo_mass": 10**13,
                    "concentration": 10,
                    "e1_mass": 0.1,
                    "e2_mass": -0.1,
                    "stellar_mass": 10e11,
                    "angular_size": 0.001,
                    "e1_light": -0.1,
                    "e2_light": 0.1,
                    "z": 0.5,
                    "mag_g": -20,
                },
            ),
        ]
        self.nfw_cluster = NFWCluster(
            deflector_dict=self.deflector_dict, subhalos_list=subhalos_list
        )

    def test_redshift(self):
        z = self.nfw_cluster.redshift
        assert self.deflector_dict["z"] == z

    def test_halo_properties(self):
        m_halo, c = self.nfw_cluster.halo_properties
        assert m_halo == self.deflector_dict["halo_mass"]
        assert c == self.deflector_dict["concentration"]

    def test_velocity_dispersion(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        vel_disp = self.nfw_cluster.velocity_dispersion(cosmo=cosmo)
        npt.assert_almost_equal(vel_disp, 1200, decimal=-1)

    def test_light_model_lenstronomy(self):
        lens_light_model_list, kwargs_lens_light = (
            self.nfw_cluster.light_model_lenstronomy(band="g")
        )
        assert len(lens_light_model_list) == 2
        assert len(kwargs_lens_light) == 2
