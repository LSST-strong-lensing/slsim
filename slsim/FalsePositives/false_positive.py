from slsim.Lenses.lens import Lens
import numpy as np


class FalsePositive(Lens):
    """Class to manage individual false positive.

    Here, false positives refer to a configuration that includes an
    elliptical galaxy at the center with blue galaxies surrounding the
    central elliptical galaxy.
    """

    def __init__(
        self,
        source_class,
        deflector_class,
        cosmo,
        los_class=None,
        include_deflector_light=True,
    ):
        """
        :param source_class: A Source class instance or list of Source class instance
        :type source_class: Source class instance from slsim.Sources.source
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
        :param cosmo: astropy.cosmology instance
        :param los_class: line of sight dictionary (optional, takes these values instead of drawing from distribution)
        :type los_class: ~LOSIndividual() class object
        """
        Lens.__init__(
            self,
            source_class=source_class,
            deflector_class=deflector_class,
            cosmo=cosmo,
            los_class=los_class,
        )
        self._include_deflector_light = include_deflector_light

    def _image_position_from_source(self, x_source, y_source, source_index):
        """Overrides the lens equation solver. For unlensed objects (on the
        lens plane), the image position is the source position.

        :return: Arrays of x and y coordinates.
        """
        return np.array([x_source]), np.array([y_source])

    def _point_source_magnification(self, source_index, extended=False):
        """Overrides the magnification calculation. For unlensed objects, the
        magnification is always 1 (flux is unchanged).

        :return: Array of magnifications (all 1.0).
        """
        # We return an array of 1.0s matching the number of "images" (which is 1 per source)
        return np.array([1.0])

    def _point_source_arrival_times(self, source_index):
        """Overrides time delay calculation.

        No lensing means no geometric or potential time delays relative
        to the source itself.
        """
        return np.array([0.0])

    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-
            normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        lens_model, kwargs_lens = self.deflector_mass_model_lenstronomy(source_index=0)
        lens_model_list = lens_model.lens_model_list
        (
            lens_light_model_list,
            kwargs_lens_light,
        ) = self.deflector.light_model_lenstronomy(band=band)

        sources, sources_kwargs = self.source_light_model_lenstronomy(band=band)

        combined_lens_light_model_list = sources["source_light_model_list"]
        combined_kwargs_lens_light = sources_kwargs["kwargs_source"]
        if self._include_deflector_light:
            combined_lens_light_model_list += lens_light_model_list
            combined_kwargs_lens_light += kwargs_lens_light

        kwargs_model = {
            "lens_light_model_list": combined_lens_light_model_list,
            "lens_model_list": lens_model_list,
        }
        if "point_source_model_list" in sources:
            kwargs_model["point_source_model_list"] = sources["point_source_model_list"]

        kwargs_source = None
        kwargs_ps = sources_kwargs["kwargs_ps"]

        kwargs_params = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": combined_kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }
        return kwargs_model, kwargs_params
