from slsim.Lenses.LensPopulation.lenses_from_catalog import LensPopCatalog
from slsim.Lenses.lens import Lens
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM


class TestLensPopCatalog(object):

    def setup_method(self):
        lens_table = Table(
            names=(
                "z_deflector",
                "z_source",
                "center_x_deflector",
                "center_y_deflector",
                "center_x_source",
                "center_y_source",
                "mag_i_light_deflector",
                "mag_i_source",
                "angular_size_light_deflector",
                "angular_size_source",
                "n_sersic_light_deflector",
                "n_sersic_source",
                "e1_light_deflector",
                "e2_light_deflector",
                "e1_source",
                "e2_source",
                "vel_disp_deflector",
                "gamma1_los",
                "gamma2_los",
                "kappa_los",
            ),
            rows=[
                (
                    0.5,
                    1.5,
                    0.01,
                    -0.01,
                    0.1,
                    -0.1,
                    17,
                    19,
                    1.5,
                    0.5,
                    1,
                    4,
                    -0.1,
                    -0.1,
                    0.1,
                    0.2,
                    250,
                    0.01,
                    -0.02,
                    0.03,
                )
            ],
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.lens_pop_catalog = LensPopCatalog(lens_catalog=lens_table, cosmo=cosmo)

        dict_list = [
            {
                "z_deflector": 0.2,
                "z_source": 2,
                "center_x_deflector": 0,
                "center_y_deflector": 0,
                "center_x_source": 0.1,
                "center_y_source": 0,
                "mag_i_light_deflector": 17,
                "mag_i_source": 22,
                "angular_size_light_deflector": 1,
                "angular_size_source": 0.5,
                "n_sersic_light_deflector": 4,
                "n_sersic_source": 1,
                "e1_light_deflector": 0,
                "e2_light_deflector": 0.2,
                "e1_source": 0.1,
                "e2_source": 0.1,
                "vel_disp_deflector": 250,
                "gamma1_los": 0.01,
                "gamma2_los": 0.01,
                "kappa_los": 0.03,
            },
        ]
        self.lens_pop_catalog2 = LensPopCatalog(lens_catalog=dict_list, cosmo=cosmo)

    def test_lens_from_table(self):

        lens = self.lens_pop_catalog.lens_from_table(index=0)
        assert isinstance(lens, Lens)

        lens = self.lens_pop_catalog.lens_from_table(index=None)
        assert isinstance(lens, Lens)

        lens = self.lens_pop_catalog2.lens_from_table(index=None)
        assert isinstance(lens, Lens)
