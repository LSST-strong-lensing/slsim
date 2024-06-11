import os
import slsim
import pandas as pd


class catalogPipeline:
    """Class for skypy configuration."""

    def __init__(self, catalog_config=None):
        """
        :param catalog_config: path to configuration file.
        :type catalog_config: string
        """
        path = os.path.dirname(slsim.__file__)
        module_path, _ = os.path.split(path)
        folder = os.path.join(module_path, catalog_config)
        self._sources = pd.read_csv(os.path.join(folder, "sources.csv"), index_col=0)
        self._deflectors = pd.read_csv(
            os.path.join(folder, "deflectors.csv"), index_col=0
        )

    @property
    def sources(self):
        """Source properties.

        :return: list of COSMODC2 Quasars
        :rtype: list of dict
        """
        return self._sources

    @property
    def deflectors(self):
        """Deflector properties.

        :return: list of OM10 galaxies
        :rtype: list of dict
        """
        return self._deflectors
