import os
from skypy.pipeline import Pipeline
import slsim
import tempfile
import slsim.Util.param_util as util


class SkyPyPipeline:
    """Class for skypy configuration."""

    def __init__(
        self,
        skypy_config=None,
        sky_area=None,
        filters=None,
        cosmo=None,
    ):
        """
        :param skypy_config: path to SkyPy configuration yaml file.
                            If None, uses 'data/SkyPy/lsst-like.yml'.
        :type skypy_config: string or None
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which galaxies are sampled.
                                Must be in units of solid angle.
        :param filters: filters for SED integration
        :type filters: list of strings or None
        :param cosmo: An instance of an astropy cosmology model
                        (e.g., FlatLambdaCDM(H0=70, Om0=0.3)).
        :type cosmo: astropy.cosmology instance or None
        """
        path = os.path.dirname(slsim.__file__)
        module_path, _ = os.path.split(path)
        if skypy_config is None:
            skypy_config = os.path.join(
                module_path, "data/SkyPy/lsst-like_triple_SF.yml"
            )
        elif skypy_config == "lsst_like_old":
            skypy_config = os.path.join(module_path, "data/SkyPy/lsst-like.yml")
        else:
            skypy_config = skypy_config

        if sky_area is None and filters is None and cosmo is None:
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, "r") as file:
                content = file.read()

            content = util.update_cosmology_in_yaml_file(cosmo=cosmo, yml_file=content)
            content = util.insert_fsky_in_yml_file(fsky=sky_area, yml_file=content)
            content = util.insert_filters_in_yaml_file(
                filters=filters, yml_file=content
            )
            content = util.unpdate_mag_key_in_yaml_file(
                filters=filters, yml_file=content
            )

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".yml"
            ) as tmp_file:
                tmp_file.write(content)

            self._pipeline = Pipeline.read(tmp_file.name)
            self._pipeline.execute()

            # Remove the temporary file after the pipeline has been executed
            os.remove(tmp_file.name)
        # TODO: note that the f_sky can not be set to large. Need to figure out
        #  how to do this properly

        # TODO: make filters work

    @property
    def blue_galaxies(self):
        """Skypy pipeline for blue galaxies.

        :return: list of blue galaxies
        :rtype: list of dict
        """
        return self._pipeline["blue"]

    @property
    def red_galaxies(self):
        """Skypy pipeline for red galaxies.

        :return: list of red galaxies
        :rtype: list of dict
        """
        return self._pipeline["red"]
