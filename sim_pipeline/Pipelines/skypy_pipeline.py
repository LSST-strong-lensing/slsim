import os
from skypy.pipeline import Pipeline
import sim_pipeline
import tempfile


class SkyPyPipeline:
    """Class for skypy configuration."""

    def __init__(self, skypy_config=None, sky_area=None, filters=None, cosmo=None):
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
        path = os.path.dirname(sim_pipeline.__file__)
        module_path, _ = os.path.split(path)
        if skypy_config is None:
            skypy_config = os.path.join(module_path, "data/SkyPy/lsst-like.yml")

        if sky_area is None and filters is None and cosmo is None:
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, "r") as file:
                content = file.read()

            if sky_area is not None:
                old_fsky = "fsky: 0.1 deg2"
                new_fsky = f"fsky: {sky_area.value} {sky_area.unit}"
                content = content.replace(old_fsky, new_fsky)

            if cosmo:
                cosmology_model_name = cosmo.__class__.__name__

                cosmology_params_list = []
                for param in cosmo.__parameters__:
                    value = getattr(cosmo, param)
                    if value is None:
                        continue
                    if isinstance(value, (int, float)):
                        cosmology_params_list.append(f"    {param}: {value}")
                    elif hasattr(value, "value"):
                        cosmology_params_list.append(f"    {param}: {value.value}")
                    else:
                        cosmology_params_list.append(f"    {param}: {value}")

                cosmology_params_str = "\n".join(cosmology_params_list)

                old_cosmo = "cosmology: !astropy.cosmology.default_cosmology.get []"
                new_cosmo = (
                    f"cosmology: !astropy.cosmology."
                    f"{cosmology_model_name}\n{cosmology_params_str}"
                )
                content = content.replace(old_cosmo, new_cosmo)

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
