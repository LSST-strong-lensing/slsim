class SLSimObject(object):
    """Class to manage image, corresponding psf, and other properties.

    Provides all the information about an image and can be used this object in
    add_object() function.
    """

    def __init__(self, image_array, psfkernel, pixelscale):
        """
        :param image_array: image in the form of numpy array
        :param psfkernel: psf kernel associated with image_array
        :param pixelscale: pixel scale in image_array
        """
        self.image_array = image_array
        self.psfkernel = psfkernel
        self.pixelscale = pixelscale

    @property
    def image(self):
        """Returns image array."""
        return self.ImageWrapper(self.image_array)

    @property
    def psf_kernel(self):
        """Returns psf kernel."""
        return self.psfkernel

    @property
    def pixel_scale(self):
        """Returns pixel scale."""
        return self.pixelscale

    class ImageWrapper:
        """Wrapper class to access the 'array' attribute directly."""

        def __init__(self, image_array):
            self.array = image_array
