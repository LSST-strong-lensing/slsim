from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy import units as u
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Sources.SourcePopulation.galaxies import (
    galaxy_projected_eccentricity,
    convert_to_slsim_convention,
    down_sample_to_dc2,
)
from astropy.table import Table
from numpy import testing as npt
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
            names=("z", "n0", "ellipticity", "angular_size", "mag_i"),
        )
        self.galaxy_list2 = Table(
            [
                [0.5, 0.5, 0.5],
                [-15.248975044343094, -15.248975044343094, -15.248975044343094],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08]
                * u.rad,
                [23, 23, 23],
                [43, 43, 43],
                [2.245543177998075, 1.9, 2.3] * u.kpc,
            ],
            names=("z", "M", "e", "angular_size", "mag_i", "a_rot", "physical_size"),
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
            extended_source_type="single_sersic",
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
            list_type="astropy_table",
            extended_source_type="double_sersic",
        )
        self.galaxies3 = Galaxies(
            galaxy_list=self.gal_list2,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            list_type="astropy_table",
            extended_source_type="double_sersic",
        )
        self.galaxies6 = Galaxies(
            galaxy_list=gal_list3,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            list_type="astropy_table",
            extended_source_type="double_sersic",
        )
        self.galaxies7 = Galaxies(
            galaxy_list=gal_list4,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            list_type="astropy_table",
            extended_source_type="double_sersic",
        )
        self.galaxies8 = Galaxies(
            galaxy_list=gal_list5,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            list_type="astropy_table",
            extended_source_type="double_sersic",
        )
        self.galaxies9 = Galaxies(
            galaxy_list=gal_list6,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            list_type="astropy_table",
            extended_source_type="double_sersic",
        )

        self.galaxies10 = Galaxies(
            galaxy_list=gal_list6,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            list_type="astropy_table",
            extended_source_type="triple",
        )
        self.galaxies11 = Galaxies(
            galaxy_list=galaxy_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            downsample_to_dc2=True,
        )

    def test_compare_downsample(self):
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
        assert self.galaxies11.n <= len(galaxy_list)

    def test_source_number(self):
        number = self.galaxies.source_number
        assert number > 0

    def test_draw_source(self):
        galaxy = self.galaxies.draw_source()
        galaxy_1 = self.galaxies4.draw_source()
        galaxy_2 = self.galaxies.draw_source(z_max=1)
        galaxy_3 = self.galaxies5.draw_source(z_max=0.4)
        galaxy_4 = self.galaxies.draw_source(z_min=0.4)
        galaxy_ind = self.galaxies.draw_source(galaxy_index=0)

        assert isinstance(galaxy, object)
        assert galaxy_1.angular_size == 4.186996407348755e-08
        assert galaxy_2.redshift < 1 + 0.002
        with pytest.raises(ValueError):
            self.galaxies5.draw_source()
        assert galaxy_3 is None
        assert galaxy_4.angular_size > 0
        assert galaxy_ind.angular_size > 0

    def test_draw_source_double_sersic(self):
        galaxy1 = self.galaxies2.draw_source()
        galaxy2 = self.galaxies3.draw_source()
        assert galaxy1.extended_source_magnitude("i") == 23
        assert galaxy2.extended_source_magnitude("i") == 23
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
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
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
            cosmo=self.cosmo,
        )
        galaxy_list = Table(
            [
                [0.5, 0.5, 0.5],
                [1, 1, 1],
                [0.1492770563596445, 0.1492770563596445, 0.1492770563596445],
                [4.186996407348755e-08, 4.186996407348755e-08, 4.186996407348755e-08],
                [23, 23, 23],
                [22, 22, 22],
            ],
            names=("z", "n0", "M", "ellipticity", "mag_i", "mag_g"),
        )
        galaxies4 = convert_to_slsim_convention(
            galaxy_catalog=galaxy_list,
            light_profile="single_sersic",
            input_catalog_type="skypy",
            source_size="Bernardi",
            cosmo=cosmo,
        )
        assert galaxies["z"][0] == 0.5
        assert galaxies["n_sersic_0"][0] == 1
        assert galaxies["ellipticity0"][0] == 0.1492770563596445
        assert galaxies2["a_rot"][0] == np.deg2rad(42)
        assert galaxies3["ellipticity"][0] == 0.1492770563596445
        npt.assert_almost_equal(
            galaxies4["angular_size"][0], 0.2795787515848128, decimal=8
        )


def test_galaxy_projected_eccentricity():
    e1, e2 = galaxy_projected_eccentricity(0)
    assert e1 == 0
    assert e2 == 0


def test_down_sample_to_dc2():
    galaxy_pop = Table(
        {
            "mag_i": np.random.uniform(18, 30, 10000),  # Magnitudes from 18 to 30
            "z": np.random.uniform(1.5, 5.0, 10000),  # Redshifts from 1.5 to 5.0
        }
    )
    sky_area = Quantity(value=1, unit="deg2")
    results = down_sample_to_dc2(galaxy_pop, sky_area)
    assert (
        len(results) == 6
    )  # downsamples in a 6 different bins and returns 6 different
    # samples
    assert min(results[0]["z"]) >= 2
    assert max(results[0]["z"]) < 2.5
    assert min(results[5]["z"]) >= 4.5
    assert max(results[5]["z"]) < 5


if __name__ == "__main__":
    pytest.main()
