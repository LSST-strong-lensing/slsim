from slsim.Deflectors.deflector_group import DeflectorGroup
from slsim.Deflectors.deflector_util import deflector_dict_from_table
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import os
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


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
        self.subhalos_table = subhalos_table
        z, center_x, center_y, kwargs_mass, kwargs_light = deflector_dict_from_table(
            table=self.halo_dict, mass_type="NFW", extended_source_type=None
        )

        kwargs_mass_list = [kwargs_mass]
        kwargs_light_list = [kwargs_light]
        center_x_deflector_list = [0]
        center_y_deflector_list = [0]

        for suhalo in subhalos_table:
            _, center_x_i, center_y_i, kwargs_mass_i, kwargs_light_i = (
                deflector_dict_from_table(
                    table=suhalo, mass_type="EPL", extended_source_type="single_sersic"
                )
            )
            kwargs_mass_list.append(kwargs_mass_i)
            kwargs_light_list.append(kwargs_light_i)
            center_x_deflector_list.append(center_x_i)
            center_y_deflector_list.append(center_y_i)
        self.deflector_group = DeflectorGroup(
            z=z,
            kwargs_mass_list=kwargs_mass_list,
            kwargs_light_list=kwargs_light_list,
            center_x_deflector_list=center_x_deflector_list,
            center_y_deflector_list=center_y_deflector_list,
            center_x=0,
            center_y=0,
        )

    def test_redshift(self):
        z = self.deflector_group.redshift
        assert self.halo_dict["z"] == z

    def test_halo_properties(self):
        halo_dict = self.deflector_group.deflector(deflector_index=0).mass_properties
        assert halo_dict["halo_mass"] == self.halo_dict["halo_mass"]
        assert halo_dict["concentration"] == self.halo_dict["concentration"]

    # TODO: deflector group does not have velocity dispersion definition
    """
    
    def test_velocity_dispersion(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        vel_disp = self.deflector_group.velocity_dispersion(cosmo=cosmo)
        npt.assert_almost_equal(vel_disp, 1200, decimal=-1)

    """

    def test_light_model_lenstronomy(self):
        lens_light_model_list, kwargs_lens_light = (
            self.deflector_group.light_model_lenstronomy(band="g")
        )
        # one for each subhalo
        assert len(lens_light_model_list) == 10
        assert len(kwargs_lens_light) == 10

    def test_mass_model_lenstronomy(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        lens_cosmo = LensCosmo(cosmo=cosmo, z_lens=self.halo_dict["z"], z_source=2.0)
        lens_mass_model_list, kwargs_lens_mass = (
            self.deflector_group.mass_model_lenstronomy(lens_cosmo=lens_cosmo)
        )
        assert len(lens_mass_model_list) == 11

    def test_stellar_mass(self):
        stellar_mass = self.deflector_group.stellar_mass
        npt.assert_almost_equal(stellar_mass / 1e11, 8.876, decimal=3)

    def test_magnitude(self):
        magnitude = self.deflector_group.magnitude(band="r")
        npt.assert_almost_equal(magnitude, 18.632, decimal=3)
