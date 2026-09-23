from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from astropy import units as u
from slsim.Util import param_util
from slsim.Sources.SourcePopulation.galaxies import Galaxies, _galaxy_size
from slsim.Sources.SourcePopulation.galaxies import (
    galaxy_projected_eccentricity,
    convert_catalog_to_source,
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
                [22, 22, 22],
            ],
            names=("z", "n0", "ellipticity", "angular_size", "mag_i", "mag_g"),
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
            size_model=None,
            catalog_type=None,
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
                "z",
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
                "z_host",
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
                "z",
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
            extended_source_type="double_sersic",
        )
        self.galaxies3 = Galaxies(
            galaxy_list=self.gal_list2,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            catalog_type="scotch",
            extended_source_type="double_sersic",
        )
        self.galaxies6 = Galaxies(
            galaxy_list=gal_list3,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            extended_source_type="double_sersic",
        )
        self.galaxies7 = Galaxies(
            galaxy_list=gal_list4,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            extended_source_type="double_sersic",
        )
        self.galaxies8 = Galaxies(
            galaxy_list=gal_list5,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            extended_source_type="double_sersic",
            catalog_type="scotch",
        )
        self.galaxies9 = Galaxies(
            galaxy_list=gal_list6,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            extended_source_type="double_sersic",
        )

        self.galaxies10 = Galaxies(
            galaxy_list=gal_list6,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            extended_source_type="triple",
        )
        self.galaxies11 = Galaxies(
            galaxy_list=galaxy_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            size_model=None,
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
        assert self.galaxies11._num_galaxies_full <= len(galaxy_list)

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

    def test_draw_galaxies(self):
        galaxies = self.galaxies11.draw_galaxies(area=Quantity(1, unit="deg2"), z_max=1)
        assert isinstance(galaxies, list)
        assert len(galaxies) >= 0
        for galaxy in galaxies:
            assert galaxy.redshift <= 1

    def test_convert_to_slsim_convention(self):
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        galaxies = convert_catalog_to_source(
            galaxy=self.gal_list[0],
            extended_source_type="double_sersic",
            catalog_type=None,
            size_model=None,
        )
        # galaxies2 = convert_catalog_to_source(
        #    galaxy=self.gal_list2[0],
        #    extended_source_type="double_sersic",
        #    catalog_type="scotch",
        # )

        # galaxies3 = convert_catalog_to_source(
        #    galaxy=self.galaxy_list2[0],
        #    extended_source_type="single_sersic",
        #    catalog_type="skypy",
        #    cosmo=self.cosmo,
        # -)
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
        galaxies4 = convert_catalog_to_source(
            galaxy=galaxy_list[0],
            extended_source_type="single_sersic",
            catalog_type="skypy",
            size_model="Bernardi",
            cosmo=cosmo,
        )
        assert galaxies["z"] == 0.5
        assert galaxies["n_sersic_0"] == 1
        e1 = np.sqrt(galaxies["e1_0"] ** 2 + galaxies["e2_0"] ** 2)
        epsilon = param_util.e2epsilon(e1)
        npt.assert_almost_equal(epsilon, 0.1492770563596445, decimal=7)
        # assert galaxies["ellipticity0"] == 0.1492770563596445
        # assert galaxies2["a_rot"] == np.deg2rad(42)
        # assert galaxies3["ellipticity"] == 0.1492770563596445
        npt.assert_almost_equal(
            galaxies4["angular_size"].value, 0.2795787515848128, decimal=8
        )


def test_galaxy_projected_eccentricity():
    e1, e2 = galaxy_projected_eccentricity(0)
    assert e1 == 0
    assert e2 == 0


def test_double_sersic_catalog_single_component_fallbacks():
    """Single-component catalog fields can seed both Sersic components."""
    common = {
        "z": 0.5,
        "w0": 0.4,
        "w1": 0.6,
        "angular_size_0": 0.2,
        "angular_size_1": 0.6,
        "n_sersic_0": 1.0,
        "n_sersic_1": 4.0,
    }

    cartesian = convert_catalog_to_source(
        {**common, "e1": 0.1, "e2": -0.2},
        extended_source_type="double_sersic",
        catalog_type=None,
    )
    assert cartesian["e1_1"] == pytest.approx(0.1)
    assert cartesian["e2_1"] == pytest.approx(-0.2)

    projected = convert_catalog_to_source(
        {**common, "ellipticity": 0.2, "a_rot": 0.0},
        extended_source_type="double_sersic",
        catalog_type=None,
    )
    assert projected["e1_1"] == pytest.approx(projected["e1_0"])
    assert projected["e2_1"] == pytest.approx(projected["e2_0"])


def test_double_sersic_catalog_component_defaults_and_validation():
    common = {
        "z": 0.5,
        "w0": 0.4,
        "w1": 0.6,
        "e1": 0.1,
        "e2": 0.0,
    }

    red_galaxy = convert_catalog_to_source(
        {
            **common,
            "angular_size_0": 0.2,
            "angular_size_1": 0.6,
            "galaxy_type": "red",
        },
        extended_source_type="double_sersic",
        catalog_type=None,
    )
    assert red_galaxy["n_sersic_0"] == pytest.approx(4.0)
    assert red_galaxy["n_sersic_1"] == pytest.approx(4.0)

    with pytest.raises(ValueError, match="component_radius_factors"):
        convert_catalog_to_source(
            {
                **common,
                "angular_size": 0.5,
                "n_sersic_0": 1.0,
                "n_sersic_1": 4.0,
                "color_gradient": {"component_radius_factors": [1.0]},
            },
            extended_source_type="double_sersic",
            catalog_type=None,
            cosmo=FlatLambdaCDM(H0=70, Om0=0.3),
        )

    with pytest.raises(
        ValueError, match="Cannot determine DoubleSersic component sizes"
    ):
        convert_catalog_to_source(
            {
                **common,
                "n_sersic_0": 1.0,
                "n_sersic_1": 4.0,
            },
            extended_source_type="double_sersic",
            catalog_type=None,
        )

    with pytest.raises(ValueError, match="component_sersic_indices"):
        convert_catalog_to_source(
            {
                **common,
                "angular_size_0": 0.2,
                "angular_size_1": 0.6,
                "color_gradient": {"component_sersic_indices": [1.0]},
            },
            extended_source_type="double_sersic",
            catalog_type=None,
        )

    with pytest.raises(ValueError, match="missing reference-band flux weights"):
        convert_catalog_to_source(
            {
                "z": 0.5,
                "e1": 0.1,
                "e2": 0.0,
                "angular_size_0": 0.2,
                "angular_size_1": 0.6,
                "n_sersic_0": 1.0,
                "n_sersic_1": 4.0,
            },
            extended_source_type="double_sersic",
            catalog_type=None,
        )

    with pytest.raises(ValueError, match="color_gradient.*must be a dictionary"):
        convert_catalog_to_source(
            {
                **common,
                "angular_size_0": 0.2,
                "angular_size_1": 0.6,
                "n_sersic_0": 1.0,
                "n_sersic_1": 4.0,
                "color_gradient": "invalid",
            },
            extended_source_type="double_sersic",
            catalog_type=None,
        )


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


def test_galaxy_size():

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = 1
    with npt.assert_raises(ValueError):
        # Bernardi size model needs mag_g in columns
        galaxy = {"z": z}
        size_model = "Bernardi"
        angular_size, physical_size = _galaxy_size(
            galaxy, size_model, catalog_type="skypy", cosmo=cosmo
        )
    galaxy = {"z": z, "physical_size": 1 * u.kpc}
    angular_size, physical_size = _galaxy_size(
        galaxy, size_model="NONE", catalog_type=None, cosmo=cosmo
    )
    npt.assert_almost_equal(physical_size, 1, decimal=5)

    galaxy = {"z": z, "physical_size": 1.0 / 1000 * u.Mpc}
    angular_size, physical_size = _galaxy_size(
        galaxy, size_model="NONE", catalog_type=None, cosmo=cosmo
    )
    npt.assert_almost_equal(physical_size, 1, decimal=5)

    galaxy = {"z": z, "angular_size": 1 * u.arcsec}
    angular_size, physical_size = _galaxy_size(
        galaxy, size_model="NONE", catalog_type=None, cosmo=cosmo
    )
    npt.assert_almost_equal(angular_size, 1, decimal=5)

    galaxy = {"z": z, "angular_size": np.pi / 3600 / 180 * u.rad}
    angular_size, physical_size = _galaxy_size(
        galaxy, size_model="NONE", catalog_type=None, cosmo=cosmo
    )
    npt.assert_almost_equal(angular_size, 1, decimal=5)

    physical_size = 1.0  # in kpc
    physical_size_mpc = physical_size / 1000.0
    angular_size_from_phys = (
        physical_size_mpc
        / cosmo.angular_diameter_distance(z).value
        * 3600
        * 180
        / np.pi
    )

    galaxy = {"z": z, "angular_size": angular_size_from_phys}
    angular_size, physical_size = _galaxy_size(
        galaxy, size_model="NONE", catalog_type=None, cosmo=cosmo
    )
    npt.assert_almost_equal(physical_size, 1, decimal=5)

    galaxy = {"z": z, "physical_size": 1}
    angular_size, physical_size = _galaxy_size(
        galaxy, size_model="NONE", catalog_type=None, cosmo=cosmo
    )
    npt.assert_almost_equal(angular_size, angular_size_from_phys, decimal=5)


if __name__ == "__main__":
    pytest.main()
