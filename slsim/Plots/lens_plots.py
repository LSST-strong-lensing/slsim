import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from slsim.image_simulation import simulate_image


class LensingPlots(object):
    """A class to create and display simulated gravitational lensing images using the
    provided configurations for the source (blue) and lens (red) galaxies."""

    def __init__(self, lens_pop, num_pix=64, observatory="LSST", **kwargs):
        """

        :param lens_pop: lens population class
        :type lens_pop: `LensPop`
        :param num_pix: number of pixels for the simulated image, default is 64
        :type num_pix: int
        :param observatory: observatory chosen
        :type observatory: str
        :param kwargs: additional keyword arguments for the bands. Eg: coadd_years
            (=10): this is the number of years corresponding to num_exposures in obs
            dict. Currently supported: 1-10.
        :type kwargs: dict
        """
        self._lens_pop = lens_pop
        self.num_pix = num_pix
        self._observatory = observatory
        self._kwargs = kwargs

    def rgb_image(self, lens_class, rgb_band_list, add_noise=True):
        """Method to generate a rgb-image with lupton_rgb color scale.

        :param lens_class: class object containing all information of the lensing system
            (e.g., Lens())
        :param rgb_band_list: list of imaging band names corresponding to r-g-b color
            map
        :param add_noise: boolean flag, set to True to add noise to the image, default
            is True
        """
        image_r = simulate_image(
            lens_class=lens_class,
            band=rgb_band_list[0],
            num_pix=self.num_pix,
            add_noise=add_noise,
            observatory=self._observatory,
            **self._kwargs
        )
        image_g = simulate_image(
            lens_class=lens_class,
            band=rgb_band_list[1],
            num_pix=self.num_pix,
            add_noise=add_noise,
            observatory=self._observatory,
            **self._kwargs
        )
        image_b = simulate_image(
            lens_class=lens_class,
            band=rgb_band_list[2],
            num_pix=self.num_pix,
            add_noise=add_noise,
            observatory=self._observatory,
            **self._kwargs
        )
        image_rgb = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image_rgb

    def plot_montage(
        self,
        rgb_band_list,
        add_noise=True,
        n_horizont=1,
        n_vertical=1,
        kwargs_lens_cut=None,
    ):
        """Method to generate and display a grid of simulated gravitational lensing
        images with or without noise.

        :param rgb_band_list: list of imaging band names corresponding to r-g-b color
            map
        :param add_noise: boolean flag, set to True to add noise to the images, default
            is True
        :param n_horizont: number of images to display horizontally, default is 1
        :param n_vertical: number of images to display vertically, default is 1
        :param kwargs_lens_cut: lens selection cuts for Lens.validity_test() function
        """
        if kwargs_lens_cut is None:
            kwargs_lens_cut = {}
        fig, axes = plt.subplots(
            n_vertical, n_horizont, figsize=(n_horizont * 3, n_vertical * 3)
        )
        for i in range(n_horizont):
            for j in range(n_vertical):
                ax = axes[j, i]
                lens_class = self._lens_pop.select_lens_at_random(**kwargs_lens_cut)
                image_rgb = self.rgb_image(
                    lens_class, rgb_band_list, add_noise=add_noise
                )
                ax.imshow(image_rgb, aspect="equal", origin="lower")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.autoscale(False)

        fig.tight_layout()
        fig.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )
        return fig, axes
