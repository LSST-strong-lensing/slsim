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
        field_galaxies=None,
    ):
        """
        :param source_class: A Source class instance or list of Source class instance
        :type source_class: Source class instance from slsim.Sources.source
        :param deflector_class: deflector instance
        :type deflector_class: Deflector class instance from slsim.Deflectors.deflector
        :param cosmo: astropy.cosmology instance
        :type cosmo: astropy.cosmology instance
        :param los_class: line of sight dictionary (optional, takes these values instead of drawing from distribution)
        :type los_class: ~LOSIndividual() class object
        :param include_deflector_light: whether to include the deflector light in the final lenstronomy kwargs output.  Default is True.
        :type include_deflector_light: bool
        :param field_galaxies: List of field galaxy instances to include in the lensing configuration.
            These contribute to the lens plane light but are not treated as mass deflectors.
            Instances should be generated via :meth:`slsim.Lenses.lens_pop.draw_field_galaxies`
            using a `slsim.Sources.SourcePopulation.Galaxies` population, based on the
            image area and redshift range to maintain a consistent number density.
            If None, no field galaxies are included.
        :type field_galaxies: list[`slsim.Sources.source.Source`] or None
        """
        Lens.__init__(
            self,
            source_class=source_class,
            deflector_class=deflector_class,
            cosmo=cosmo,
            los_class=los_class,
            field_galaxies=field_galaxies,
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
        to the unlensed light.
        """
        return np.array([0.0])

    def lenstronomy_kwargs(self, band=None, time = None):
        """Generates lenstronomy dictionary conventions for the class object.

        :param band: imaging band, if =None, will result in un-
            normalized amplitudes
        :type band: string or None
        :param time: observation time, used to calculate the luminosity of transient sources at the time of observation (optional, default is None, which will use the peak luminosity)
        :type time: float or None
        """
        lens_model, kwargs_lens = self.deflector_mass_model_lenstronomy(source_index=0)
        lens_model_list = lens_model.lens_model_list
        (
            lens_light_model_list,
            kwargs_lens_light,
        ) = self.deflector.light_model_lenstronomy(band=band)

        # turn off lensing
        for i in range(self.source_number):
            if self.source(i).point_source is not None:
                self.source(i).point_source.lensed = False
            if self.source(i).extended_source is not None:
                self.source(i).extended_source.lensed = False

        sources, sources_kwargs = self.source_light_model_lenstronomy(band=band, time=time)
        combined_lens_light_model_list = sources["source_light_model_list"]
        combined_kwargs_lens_light = sources_kwargs["kwargs_source"]

        # field galaxies
        if self._field_galaxies is not None:
            field_galaxies_lens_model_list, kwargs_field_galaxies = (
                self.field_galaxy_light_model_lenstronomy(band=band)
            )
            combined_lens_light_model_list += field_galaxies_lens_model_list
            combined_kwargs_lens_light += kwargs_field_galaxies

        # to include the deflector light
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
