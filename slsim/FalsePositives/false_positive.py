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
        combined_lens_light_model_list = (
            lens_light_model_list + sources["source_light_model_list"]
        )
        combined_kwargs_lens_light = kwargs_lens_light + sources_kwargs["kwargs_source"]

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
