from slsim.slsim_object import SLSimObject
import numpy as np
import pytest


@pytest.fixture
def test_SLSimObject():
    image_array = np.array([[1, 2], [3, 4]])
    psfkernel = np.array([[0.1, 0.2], [0.3, 0.4]])
    pixelscale = 0.05
    slsim_object = SLSimObject(image_array, psfkernel, pixelscale)
    return slsim_object


def test_image_property(test_SLSimObject):
    image_result = test_SLSimObject.image.array
    assert np.shape(image_result) == (2, 2)


def test_psf_kernel_property(test_SLSimObject):
    psf_result = test_SLSimObject.psf_kernel
    assert np.shape(psf_result) == (2, 2)


def test_pixel_scale_property(test_SLSimObject):
    scale_result = test_SLSimObject.pixel_scale
    assert scale_result == 0.05


if __name__ == "__main__":
    pytest.main()
