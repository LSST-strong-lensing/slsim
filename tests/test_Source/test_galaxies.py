from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.galaxies import Galaxies
from slsim.Sources.galaxies import (
    galaxy_projected_eccentricity,
    convert_to_slsim_convention,
)
from astropy.table import Table
import pytest
import numpy as np


class TestGalaxies(object):
    def setup_method(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        galaxy_list = Table(
            [
                [0.5, 0.5, 0.5],
                [-15.248975044343094, -15.248975044343094, -15.248975044343094],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08],
                [23, 23, 23],
            ],
            names=("z", "n0", "M", "ellipticity", "mag_i"),
        )
        self.galaxy_list2 = Table(
            [
                [0.5, 0.5, 0.5],
                [-15.248975044343094, -15.248975044343094, -15.248975044343094],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08],
                [23, 23, 23],
                [43, 43, 43],
            ],
            names=("z", "M", "e", "angular_size", "mag_i", "a_rot"),
        )

        galaxy_list3 = Table(
            [
                [0.5, 0.5, 0.5],
                [-15.248975044343094, -15.248975044343094, -15.248975044343094],
                [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08],
                [23, 23, 23],
                [43, 43, 43],
            ],
            names=("z", "M", "angular_size", "mag_i", "a_rot"),
        )
        self.galaxies = Galaxies(
            galaxy_list=galaxy_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
        )
        self.galaxies4 = Galaxies(
            galaxy_list=self.galaxy_list2,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
        )
        self.galaxies5 = Galaxies(
            galaxy_list=galaxy_list3,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
        )

        self.gal_list = Table(
            [
                [0.5, 0.5, 0.5],
                [1, 1, 1],
                [4, 4, 4],
                [0.28254256, 0.28254256, 0.28254256],
                [0.17744173, 0.17744173, 0.17744173],
                [0.2091449, 0.2091449, 0.2091449],
                [0.14535427, 0.14535427, 0.14535427],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [0.0994047812516054, 0.0994047812516054, 0.0994047812516054],
                [0.9724, 0.9724, 0.9724],
                [0.0276, 0.0276, 0.0276],
                [23, 23, 23],
                [-0.1781825499400982, -0.1781825499400982, -0.1781825499400982],
                [0.2665196551390636, 0.2665196551390636, 0.2665196551390636],
            ],
            names=(
                "z_host",
                "n0",
                "n1",
                "a0",
                "a1",
                "b0",
                "b1",
                "e0",
                "e1",
                "w0",
                "w1",
                "mag_i",
                "ra_off",
                "dec_off",
            ),
        )

        self.gal_list2 = Table(
            [
                [0.5, 0.5, 0.5],
                [0.28254256, 0.28254256, 0.28254256],
                [0.17744173, 0.17744173, 0.17744173],
                [0.2091449, 0.2091449, 0.2091449],
                [0.14535427, 0.14535427, 0.14535427],
                [0.9724, 0.9724, 0.9724],
                [0.0276, 0.0276, 0.0276],
                [23, 23, 23],
                [-0.1781825499400982, -0.1781825499400982, -0.1781825499400982],
                [0.2665196551390636, 0.2665196551390636, 0.2665196551390636],
                [42, 42, 42],
            ],
            names=(
                "z",
                "a0",
                "a1",
                "b0",
                "b1",
                "w0",
                "w1",
                "mag_i",
                "ra_off",
                "dec_off",
                "a_rot",
            ),
        )

        gal_list3 = Table(
            [
                [0.5, 0.5, 0.5],
                [0.9724, 0.9724, 0.9724],
                [0.0276, 0.0276, 0.0276],
                [23, 23, 23],
                [-0.1781825499400982, -0.1781825499400982, -0.1781825499400982],
                [0.2665196551390636, 0.2665196551390636, 0.2665196551390636],
                [42, 42, 42],
            ],
            names=("z", "w0", "w1", "mag_i", "ra_off", "dec_off", "a_rot"),
        )

        gal_list4 = Table(
            [
                [0.5, 0.5, 0.5],
                [0.28254256, 0.28254256, 0.28254256],
                [0.2091449, 0.2091449, 0.2091449],
                [0.9724, 0.9724, 0.9724],
                [0.0276, 0.0276, 0.0276],
                [23, 23, 23],
                [-0.1781825499400982, -0.1781825499400982, -0.1781825499400982],
                [0.2665196551390636, 0.2665196551390636, 0.2665196551390636],
                [42, 42, 42],
            ],
            names=("z", "a0", "b0", "w0", "w1", "mag_i", "ra_off", "dec_off", "a_rot"),
        )
        gal_list5 = Table(
            [
                [0.5, 0.5, 0.5],
                [1, 1, 1],
                [4, 4, 4],
                [0.17744173, 0.17744173, 0.17744173],
                [0.14535427, 0.14535427, 0.14535427],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [0.0994047812516054, 0.0994047812516054, 0.0994047812516054],
                [0.9724, 0.9724, 0.9724],
                [0.0276, 0.0276, 0.0276],
                [23, 23, 23],
                [-0.1781825499400982, -0.1781825499400982, -0.1781825499400982],
                [0.2665196551390636, 0.2665196551390636, 0.2665196551390636],
            ],
            names=(
                "z_host",
                "n0",
                "n1",
                "a1",
                "b1",
                "e0",
                "e1",
                "w0",
                "w1",
                "mag_i",
                "ra_off",
                "dec_off",
            ),
        )
        gal_list6 = Table(
            [
                [0.5, 0.5, 0.5],
                [1, 1, 1],
                [4, 4, 4],
                [0.28254256, 0.28254256, 0.28254256],
                [0.2091449, 0.2091449, 0.2091449],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [0.0994047812516054, 0.0994047812516054, 0.0994047812516054],
                [0.9724, 0.9724, 0.9724],
                [0.0276, 0.0276, 0.0276],
                [23, 23, 23],
                [-0.1781825499400982, -0.1781825499400982, -0.1781825499400982],
                [0.2665196551390636, 0.2665196551390636, 0.2665196551390636],
            ],
            names=(
                "z_host",
                "n0",
                "n1",
                "a0",
                "b0",
                "e0",
                "e1",
                "w0",
                "w1",
                "mag_i",
                "ra_off",
                "dec_off",
            ),
        )
        self.galaxies2 = Galaxies(
            galaxy_list=self.gal_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="double_sersic",
            list_type="astropy_table",
        )
        self.galaxies3 = Galaxies(
            galaxy_list=self.gal_list2,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="double_sersic",
            list_type="astropy_table",
        )
        self.galaxies6 = Galaxies(
            galaxy_list=gal_list3,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="double_sersic",
            list_type="astropy_table",
        )
        self.galaxies7 = Galaxies(
            galaxy_list=gal_list4,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="double_sersic",
            list_type="astropy_table",
        )
        self.galaxies8 = Galaxies(
            galaxy_list=gal_list5,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="double_sersic",
            list_type="astropy_table",
        )
        self.galaxies9 = Galaxies(
            galaxy_list=gal_list6,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="double_sersic",
            list_type="astropy_table",
        )
        self.galaxies10 = Galaxies(
            galaxy_list=gal_list6,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            light_profile="triple",
            list_type="astropy_table",
        )

    def test_source_number(self):
        number = self.galaxies.source_number()
        assert number > 0

    def test_draw_source(self):
        galaxy = self.galaxies.draw_source()
        galaxy_1 = self.galaxies4.draw_source()
        assert len(galaxy) > 0
        assert galaxy_1["n_sersic"] == 1
        with pytest.raises(ValueError):
            self.galaxies5.draw_source()

    def test_draw_source_double_sersic(self):
        galaxy1 = self.galaxies2.draw_source()
        galaxy2 = self.galaxies3.draw_source()
        assert galaxy1["n_sersic_0"] == 1
        assert galaxy1["n_sersic_1"] == 4
        assert galaxy2["n_sersic_0"] == 1
        assert galaxy2["n_sersic_1"] == 4
        with pytest.raises(ValueError):
            self.galaxies6.draw_source()
        with pytest.raises(ValueError):
            self.galaxies7.draw_source()
        with pytest.raises(ValueError):
            self.galaxies8.draw_source()
        with pytest.raises(ValueError):
            self.galaxies9.draw_source()
        with pytest.raises(ValueError):
            self.galaxies10.draw_source()

    def test_convert_to_slsim_convention(self):
        galaxies = convert_to_slsim_convention(
            galaxy_catalog=self.gal_list,
            light_profile="double_sersic",
            input_catalog_type=None,
        )
        galaxies2 = convert_to_slsim_convention(
            galaxy_catalog=self.gal_list2,
            light_profile="double_sersic",
            input_catalog_type="scotch",
        )
        galaxies3 = convert_to_slsim_convention(
            galaxy_catalog=self.galaxy_list2,
            light_profile="single_sersic",
            input_catalog_type="skypy",
        )
        assert galaxies["z"][0] == 0.5
        assert galaxies["n_sersic_0"][0] == 1
        assert galaxies["ellipticity0"][0] == 0.1492770563596445
        assert galaxies2["a_rot"][0] == np.deg2rad(42)
        assert galaxies3["ellipticity"][0] == 0.1492770563596445


def test_galaxy_projected_eccentricity():
    e1, e2 = galaxy_projected_eccentricity(0)
    assert e1 == 0
    assert e2 == 0


if __name__ == "__main__":
    pytest.main()
