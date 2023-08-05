import os
from skypy.pipeline import Pipeline
import sim_pipeline
import tempfile


class HalosSkyPyPipeline:
    def __init__(self, skypy_config=None, sky_area=None, filters=None):
        """
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string or None
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area over which halos are sampled. Must be in units of solid angle.
        :param filters: filters for SED integration
        :type filters: list of strings or None
        """
        path = os.path.dirname(sim_pipeline.__file__)
        module_path, _ = os.path.split(path)
        if skypy_config is None:
            skypy_config = os.path.join(module_path, 'data/SkyPy/halo.yml')

        if sky_area is None and filters is None:
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, 'r') as file:
                content = file.read()

            if sky_area is not None:
                old_fsky = "fsky: 0.0001 deg2"
                new_fsky = f"fsky: %s deg2" % sky_area
                content = content.replace(old_fsky, new_fsky)

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml') as tmp_file:
                tmp_file.write(content)

            self._pipeline = Pipeline.read(tmp_file.name)
            self._pipeline.execute()

            # Remove the temporary file after the pipeline has been executed
            os.remove(tmp_file.name)
        #TODO: note that the f_sky can not be set to large. Need to figure out how to do this properly
        # for LSST simulations (10^5 deg^2)

    @property
    def halos(self):
        """
        skypy pipeline for blue galaxies

        :return: list of blue galaxies
        :rtype: list of dict
        """
        return self._pipeline['halos']

