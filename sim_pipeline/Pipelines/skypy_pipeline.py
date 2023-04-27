import os
from skypy.pipeline import Pipeline
import sim_pipeline


class SkyPyPipeline(object):
    """
    running of skypy pipeline

    """
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
            # with open(skypy_config, 'r') as file:
            #    configuration = yaml.load(file, Loader=yaml.Loader)
            #    fsky_yaml = configuration['fsky']
            #    assert fsky_yaml == f_sky  # TODO: make sky fraction a variable input outside of yaml file

        self._pipeline = Pipeline.read(skypy_config)
        self._pipeline.execute()

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


