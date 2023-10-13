import numpy as np
import os
from slsim.Util.param_util import (
    epsilon2e,
    e2epsilon,
    random_ra_dec,
    convolved_image,
)


def test_epsilon2e():
    e = epsilon2e(0)
    assert e == 0


def test_e2epsilon():
    ep = e2epsilon(0)
    assert ep == 0


def test_random_ra_dec():
    ra, dec = random_ra_dec(ra_min=30, ra_max=62, dec_min=-63, dec_max=-36, n=50)
    assert len(ra) == 50


def test_convolved_image():
    path = os.path.dirname(__file__)
    image = np.load(os.path.join(path, "TestData/image.npy"))
    psf = np.load(os.path.join(path, "TestData/psf_kernels_for_deflector.npy"))
    c_image = convolved_image(image, psf)
    c_image_1 = convolved_image(image, psf, type="grid")
    assert c_image.shape[0] == 101
    assert c_image_1.shape[0] == 101
