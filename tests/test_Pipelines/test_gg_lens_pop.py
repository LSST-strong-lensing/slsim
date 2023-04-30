import pytest
from sim_pipeline.gg_lens_pop import GGLensPop

class TestGGLensPop:

    @pytest.fixture(scope="class")
    def gg_lens_pop_instance(self):
        return GGLensPop(f_sky=0.1)

    def test_num_lenses_and_sources(self, gg_lens_pop_instance):
        num_lenses = gg_lens_pop_instance.get_num_lenses()
        num_sources = gg_lens_pop_instance.get_num_sources()

        assert 5800 <= num_lenses <= 6600, f"Expected num_lenses to be between 5800 and 6600, but got {num_lenses}"
        assert 1090000 <= num_sources <= 1110000, f"Expected num_sources to be between 1090000 and 1110000, but got {num_sources}"

if __name__ == "__main__":
    pytest.main()
