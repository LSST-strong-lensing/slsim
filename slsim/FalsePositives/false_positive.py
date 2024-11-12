import numpy as np
from slsim.lens import Lens

class FalsePositive(Lens):
    """Class to manage individual false positive."""

    def __init__(
        self,
        source_class,
        deflector_class,
        cosmo,
        test_area=4 * np.pi,
        los_config=None,
        los_dict=None,
    ):
        """
        :param source_class: A Source class instance or list of Source class instance
        :type source_class: Source class instance from slsim.Sources.source
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
        :param cosmo: astropy.cosmology instance
        :param test_area: area of disk around one lensing galaxies to be investigated
            on (in arc-seconds^2).
        :param los_config: LOSConfig instance which manages line-of-sight (LOS) effects
         and Gaussian mixture models in a simulation or analysis context.
        :param los_dict: line of sight dictionary (optional, takes these values instead
         of drawing from distribution) Takes "gamma" = [gamma1, gamma2] and
         "kappa" = kappa as entries
        :type los_dict: dict
        """
        Lens.__init__(self,
            source_class=source_class,
            deflector_class=deflector_class,
            cosmo=cosmo,
            test_area=test_area,
            los_config=los_config,
            los_dict=los_dict,
            )

    def lenstronomy_kwargs(self, band=None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-normalized amplitudes
        :type band: string or None
        :return: lenstronomy model and parameter conventions
        """
        lens_mass_model_list, kwargs_lens = self.deflector_mass_model_lenstronomy()
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
            "lens_model_list": lens_mass_model_list,
        }

        kwargs_source = None
        kwargs_ps = sources_kwargs["kwargs_ps"]

        kwargs_params = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": combined_kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }

        return kwargs_model, kwargs_params