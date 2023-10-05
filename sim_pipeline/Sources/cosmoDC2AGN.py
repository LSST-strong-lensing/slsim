import pandas as pd
from sim_pipeline.Sources.source_base import SourceBase


class cosmoDC2AGN(SourceBase):
    def __init__(self, source_file, cosmo, sky_area):
        """
        :param source_file: csv file containing source properties
        :param cosmo: cosmology used
        :param sky_area:
        """
        super().__init__(cosmo, sky_area)
        self.sources = pd.read_csv(source_file)

    def source_number(self):
        return super().source_number()
