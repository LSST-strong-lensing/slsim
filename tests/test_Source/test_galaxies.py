from astropy.cosmology import FlatLambdaCDM
from slsim.Pipelines.skypy_pipeline import SkyPyPipeline
from astropy.units import Quantity
from slsim.Sources.galaxies import Galaxies
from slsim.Sources.galaxies import galaxy_projected_eccentricity
from astropy.table import Table
import pytest


class TestGalaxies(object):
    def setup_method(self):
        sky_area = Quantity(value=0.1, unit="deg2")
        pipeline = SkyPyPipeline(skypy_config=None, sky_area=sky_area, filters=None)
        self.galaxy_list = pipeline.red_galaxies
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.galaxies = Galaxies(
            galaxy_list=self.galaxy_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
        )

        gal_list = Table(
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

        gal_list2 = Table(
            [
                [0.5, 0.5, 0.5],
                [1, 1, 1],
                [4, 4, 4],
                [0.28254256, 0.28254256, 0.28254256],
                [0.17744173, 0.17744173, 0.17744173],
                [0.2091449, 0.2091449, 0.2091449],
                [0.14535427, 0.14535427, 0.14535427],
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
                "w0",
                "w1",
                "mag_i",
                "ra_off",
                "dec_off",
            ),
        )
        self.galaxies2 = Galaxies(
            galaxy_list=gal_list,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            sersic_profile="double",
            list_type="astropy_table",
        )
        self.galaxies3 = Galaxies(
            galaxy_list=gal_list2,
            kwargs_cut={},
            cosmo=self.cosmo,
            sky_area=sky_area,
            sersic_profile="double",
            list_type="astropy_table",
        )

    def test_source_number(self):
        number = self.galaxies.source_number()
        assert number > 0

    def test_draw_source(self):
        galaxy = self.galaxies.draw_source()
        assert len(galaxy) > 0

    def test_draw_source_double_sersic(self):
        galaxy1 = self.galaxies2.draw_source()
        galaxy2 = self.galaxies3.draw_source()
        assert galaxy1["n_sersic_0"] == 1
        assert galaxy1["n_sersic_1"] == 4
        assert galaxy2["n_sersic_0"] == 1
        assert galaxy2["n_sersic_1"] == 4


def test_galaxy_projected_eccentricity():
    e1, e2 = galaxy_projected_eccentricity(0)
    assert e1 == 0
    assert e2 == 0


if __name__ == "__main__":
    pytest.main()
