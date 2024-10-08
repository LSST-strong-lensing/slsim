from slsim.Sources.source_pop_base import SourcePopBase


class cosmoDC2AGN(SourcePopBase):
    def __init__(self, source_input, cosmo, sky_area):
        """
        :param source_input: dict-like containing source properties
        :param cosmo: cosmology used
        :param sky_area:
        """
        self.source_table = source_input
        super().__init__(cosmo, sky_area)

    @property
    def source_number(self):
        """Number of sources registered (within given area on the sky)

        :return: number of sources
        """
        return len(self.source_table)

    def draw_source(self, index):
        """Choose source at random.

        :param index: id number of lens and source
        :return: dictionary of source
        """
        self._chosen_source = dict(self.source_table.loc[index])
        return self._chosen_source
