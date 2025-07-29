from slsim.Util.param_util import draw_coord_in_circle
import numpy.testing as npt
import numpy as np


def test_draw_coord_in_circle():
    np.random.seed(42)
    ra, dec = draw_coord_in_circle(area=2, size=1000)
    npt.assert_almost_equal(np.mean(ra), 0, decimal=2)

    ra, dec = draw_coord_in_circle(area=2, size=1)
    assert isinstance(ra, float)
