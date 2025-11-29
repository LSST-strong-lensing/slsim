import os
from skypy.pipeline import Pipeline
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
        z_min=None,
        z_max=None,
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
        :z_min: minimum redshift of the galaxy catalog to be simulated.
        :type z_min: float or None
        :z_max: maximum redshift of the galaxy catalog to be simulated.
         If one passes u-band filter, z_max should be <= 4.09 to avoid
         issues with skypy SED templates.
        :type z_max: float or None
        """
        # path = os.path.dirname(slsim.__file__)

        path = os.path.dirname(os.path.abspath(__file__))
        path, _ = os.path.split(path)
        module_path, _ = os.path.split(path)
        if skypy_config is None:
            skypy_config = os.path.join(
                module_path, "data/SkyPy/lsst-like_triple_SF.yml"
            )
        elif skypy_config == "lsst_like_old":
            skypy_config = os.path.join(module_path, "data/SkyPy/lsst-like.yml")
        else:
            skypy_config = skypy_config

        if sky_area is None and filters is None and cosmo is None and z_min is None:
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, "r") as file:
                content = file.read()

            if sky_area is not None:
                old_fsky = "fsky: 0.1 deg2"
                new_fsky = f"fsky: {sky_area.value} {sky_area.unit}"
                content = content.replace(old_fsky, new_fsky)
            if z_min is not None and z_max is not None:
                old_zrange = "!numpy.arange [0.0, 5.01, 0.01]"
                new_zrange = f"!numpy.arange [{z_min}, {z_max}, {0.01}]"
                content = content.replace(old_zrange, new_zrange)

            if filters is not None:
                filters_mag = [f"mag_{f}" for f in filters]
                old_filter_name = "mag_g, mag_r, mag_i, mag_z, mag_y"
                new_filters_name = f"{filters_mag}".strip("[]").replace("'", "")
                old_filters = "filters: ['lsst2016-g', 'lsst2016-r', 'lsst2016-i', 'lsst2016-z', 'lsst2016-y']"

                new_filters = [f.replace("mag_", "lsst2016-") for f in filters_mag]
                new_filters = f"filters: {new_filters}"

                content = content.replace(old_filters, new_filters)
                content = content.replace(old_filter_name, new_filters_name)

            content = util.update_cosmology_in_yaml_file(cosmo=cosmo, yml_file=content)

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
