import os
from skypy.pipeline import Pipeline
import sim_pipeline
import tempfile

class SkyPyPipeline:
    def __init__(self, skypy_config=None, f_sky=0.1):
        """
        :param skypy_config: path to SkyPy configuration yaml file
        :type skypy_config: string or None
        :param f_sky: sky area (in deg^2)
        :type f_sky: float
        """
        if skypy_config is None:
            path = os.path.dirname(sim_pipeline.__file__)
            module_path, _ = os.path.split(path)
            skypy_config = os.path.join(module_path, 'data/SkyPy/lsst-like.yml')  # read the file

        if f_sky == 0.1:
            self._pipeline = Pipeline.read(skypy_config)
            self._pipeline.execute()
        else:
            with open(skypy_config, 'r') as file:
                content = file.read()

            old_fsky = "fsky: 0.1 deg2"
            new_fsky = f"fsky: {f_sky} deg2"
            new_content = content.replace(old_fsky, new_fsky)

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml') as tmp_file:
                tmp_file.write(new_content)

            self._pipeline = Pipeline.read(tmp_file.name)
            self._pipeline.execute()

            # Remove the temporary file after the pipeline has been executed
            os.remove(tmp_file.name)
        #TODO: note that the f_sky can not be set to large. Need to figure out how to do this properly
        # for LSST simulations (10^5 deg^2)

    @property
    def blue_galaxies(self):
        """
        skypy pipeline for blue galaxies

        :return: list of blue galaxies
        :rtype: list of dict
        """
        return self._pipeline['blue']

    @property
    def red_galaxies(self):
        """
        skypy pipeline for red galaxies

        :return: list of red galaxies
        :rtype: list of dict
        """
        return self._pipeline['red']
