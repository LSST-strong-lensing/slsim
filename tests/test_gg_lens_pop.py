import pytest
from sim_pipeline.gg_lens_pop import GGLensPop, draw_test_area
import numpy as np

class TestGGLensPop:

    @pytest.fixture(scope="class")
    def gg_lens_pop_instance(self):
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        from astropy.units import Quantity
        sky_area = Quantity(value=0.1, unit='deg2')
        return GGLensPop(sky_area=sky_area, cosmo=cosmo)

    def test_num_lenses_and_sources(self, gg_lens_pop_instance):
        num_lenses = gg_lens_pop_instance.get_num_lenses()
        num_sources = gg_lens_pop_instance.get_num_sources()

        assert 1000 <= num_lenses <= 6600, f"Expected num_lenses to be between 5800 and 6600, but got {num_lenses}"
        assert 100000 <= num_sources <= 500000, f"Expected num_sources to be between 1090000 and 1110000, but got {num_sources}"
        # assert 1 == 0

    def test_num_sources_tested_and_test_area(self, gg_lens_pop_instance):
        lens = gg_lens_pop_instance._lens_galaxies.draw_deflector()
        test_area = draw_test_area(deflector=lens)
        assert 0.01 < test_area < 100 * np.pi, f"Expected test_area to be between 0.1 and 100*pi, but got {test_area}"
        num_sources_range = gg_lens_pop_instance.get_num_sources_tested(testarea=test_area)
        assert 0 <= num_sources_range <= 50, f"Expected num_sources_range to be between 0 and 50, but got {num_sources_range}"


if __name__ == "__main__":
    pytest.main()
